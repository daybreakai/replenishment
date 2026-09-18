#!/usr/bin/env python3
"""Assemble a StandardSimulationRow-shaped CSV from the real
t_outbound_shipment/t_product/t_vendor_lead_time tables, instead of
requiring a client to already have a single StandardSimulationRow-shaped
table (which no real client catalog actually has -- see
~/.claude/projects/-Users-jackrodenberg/memory/project_inventory_data_model_gap.md).
Costs are derived from real unit_cost per the confirmed business
parameters (project_replenishment_cost_assumptions.md), not invented flat
scalars. forecast is a naive trailing moving average -- t_forecast's real
p10/p50/p90 quantiles aren't wired in; confirmed with the user as an
acceptable simplification for now. See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from load_from_databricks import run_query  # noqa: E402

# Confirmed 2026-09-18 -- see project_replenishment_cost_assumptions.md.
# Override via CLI flags for a different real run; don't silently pick new
# numbers here.
DEFAULT_HOLDING_COST_PERCENT = 0.15
DEFAULT_STOCKOUT_COST_MULTIPLIER = 0.40
DEFAULT_ORDER_COST = 0.0
DEFAULT_FORECAST_WINDOW = 3


def _vlog(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[verbose] {msg}", file=sys.stderr)


def _fetch(catalog: str, schema: str, profile: str, verbose: bool = False) -> tuple[list[dict], dict[str, float], dict[str, int]]:
    queries = {
        "t_outbound_shipment": f"SELECT product_id, actual_ship_date, shipped_qty FROM {catalog}.{schema}.t_outbound_shipment",
        "t_product": f"SELECT id AS product_id, unit_cost FROM {catalog}.{schema}.t_product",
        "t_vendor_lead_time": f"SELECT product_id, lead_time_days FROM {catalog}.{schema}.t_vendor_lead_time",
    }
    results = {}
    for table, sql in queries.items():
        _vlog(verbose, f"querying {table}: {sql}")
        results[table] = run_query(sql, profile)
        _vlog(verbose, f"{table} -> {len(results[table])} row(s)")

    unit_cost_by_id = {r["product_id"]: float(r["unit_cost"]) for r in results["t_product"] if r["unit_cost"] is not None}
    lead_time_by_id = {r["product_id"]: int(float(r["lead_time_days"])) for r in results["t_vendor_lead_time"] if r["lead_time_days"] is not None}
    _vlog(verbose, f"unit_cost_by_id: {unit_cost_by_id}")
    _vlog(verbose, f"lead_time_by_id (days): {lead_time_by_id}")
    return results["t_outbound_shipment"], unit_cost_by_id, lead_time_by_id


def derive_rows(
    catalog: str, schema: str, profile: str, *,
    holding_cost_percent: float = DEFAULT_HOLDING_COST_PERCENT,
    stockout_cost_multiplier: float = DEFAULT_STOCKOUT_COST_MULTIPLIER,
    order_cost: float = DEFAULT_ORDER_COST,
    forecast_window: int = DEFAULT_FORECAST_WINDOW,
    verbose: bool = False,
) -> list[dict]:
    shipments, unit_cost_by_id, lead_time_by_id = _fetch(catalog, schema, profile, verbose=verbose)

    by_product: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in shipments:
        by_product[r["product_id"]].append((r["actual_ship_date"], float(r["shipped_qty"] or 0.0)))

    missing_cost = sorted(set(by_product) - set(unit_cost_by_id))
    if missing_cost:
        raise ValueError(f"{len(missing_cost)} shipped product(s) have no usable t_product.unit_cost: {missing_cost[:5]}")
    missing_lead_time = sorted(set(by_product) - set(lead_time_by_id))
    if missing_lead_time:
        raise ValueError(f"{len(missing_lead_time)} shipped product(s) have no usable t_vendor_lead_time.lead_time_days: {missing_lead_time[:5]}")

    rows = []
    for pid, series in by_product.items():
        series.sort(key=lambda t: t[0])
        unit_cost = unit_cost_by_id[pid]
        lead_time_months = max(1, round(lead_time_by_id[pid] / 30))  # monthly-grain shipment data
        holding = round(unit_cost * holding_cost_percent, 4)
        stockout = round(unit_cost * stockout_cost_multiplier, 4)
        _vlog(verbose, f"{pid}: unit_cost={unit_cost} -> holding={holding} stockout={stockout} lead_time_months={lead_time_months}")
        demands = [round(q) for _, q in series]
        for i, (ds, _) in enumerate(series):
            window = demands[max(0, i - forecast_window):i] or [demands[i]]
            forecast = round(sum(window) / len(window))
            rows.append({
                "unique_id": pid, "ds": ds, "demand": demands[i], "forecast": forecast, "actuals": demands[i],
                "holding_cost_per_unit": holding, "stockout_cost_per_unit": stockout,
                "order_cost_per_order": order_cost, "lead_time": lead_time_months,
                "initial_on_hand": round(sum(demands) / len(demands)), "current_stock": round(sum(demands) / len(demands)),
            })
    return rows


def write_rows_to_csv(rows: list[dict], out: str) -> None:
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Never inferred -- ask the user which catalog")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--profile", required=True, help="Databricks CLI profile -- never guess/default this")
    parser.add_argument("--holding-cost-percent", type=float, default=DEFAULT_HOLDING_COST_PERCENT)
    parser.add_argument("--stockout-cost-multiplier", type=float, default=DEFAULT_STOCKOUT_COST_MULTIPLIER)
    parser.add_argument("--order-cost", type=float, default=DEFAULT_ORDER_COST)
    parser.add_argument("--forecast-window", type=int, default=DEFAULT_FORECAST_WINDOW, help="Trailing moving-average window, in periods")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verbose", action="store_true", help="Print each SQL query, row count, and per-item cost derivation to stderr as it happens")
    args = parser.parse_args()
    try:
        rows = derive_rows(
            args.catalog, args.schema, args.profile,
            holding_cost_percent=args.holding_cost_percent, stockout_cost_multiplier=args.stockout_cost_multiplier,
            order_cost=args.order_cost, forecast_window=args.forecast_window, verbose=args.verbose,
        )
        if not rows:
            raise ValueError("No rows derived -- check t_outbound_shipment has data for this catalog/schema.")
        write_rows_to_csv(rows, args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
