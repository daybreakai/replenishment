#!/usr/bin/env python3
"""Orchestrate a segmentation run: validate the request, pull the required
tables, compute the requested strategies, write the SQL audit file, and
(unless --dry-run) append-write results to <catalog>.<schema>.t_segmentation.
See ../SKILL.md.

Imports run_query() from optimize-replenishment-cost/scripts/load_from_databricks.py
instead of re-shelling to the Databricks CLI itself, and
replenishment.classification.classify_demand (via segmentation_strategies.py)
for ADI/CV2 instead of reimplementing Syntetos-Boylan thresholds a second time.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_ROOT / ".claude" / "skills" / "optimize-replenishment-cost" / "scripts"))

from build_segmentation_sql import (  # noqa: E402
    _validate_customer, build_pull_statements, build_write_back_sql, format_pull_sql_for_audit, write_audit_file,
)
from load_from_databricks import run_query  # noqa: E402
from segmentation_strategies import abc_bucket, abc_xyz_matrix, adi_cv2_class, xyz_bucket  # noqa: E402
from validate_segmentation import validate  # noqa: E402

DEFAULT_TABLES = ["t_outbound_shipment", "t_product"]
_REVENUE_STRATEGIES = {"abc_revenue", "xyz_variability", "abc_xyz_matrix", "abc_revenue_by_hierarchy"}


def _period_key(ds) -> str:
    return str(ds)[:7]  # month bucket -- shipment history here is monthly-grain


def _shipped_qty(r: dict) -> float:
    # The Databricks CLI's JSON output stringifies every column, including
    # numerics (confirmed live: shipped_qty comes back as "211.9", not
    # 211.9) -- `r.get("shipped_qty") or 0.0` alone leaves it a string and
    # a later `qty * unit_cost` crashes. Cast once, here, for every reader.
    qty = r.get("shipped_qty")
    return float(qty) if qty is not None else 0.0


def _vlog(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[verbose] {msg}", file=sys.stderr)


def fetch_rows(catalog: str, schema: str, tables: list[str], profile: str, verbose: bool = False) -> dict[str, list[dict]]:
    statements = build_pull_statements(catalog, schema, tables)
    rows_by_table = {}
    for table, sql in statements.items():
        _vlog(verbose, f"querying {table}: {sql}")
        rows = run_query(sql, profile)
        _vlog(verbose, f"{table} -> {len(rows)} row(s)")
        rows_by_table[table] = rows
    return rows_by_table


def compute_segments(rows_by_table: dict[str, list[dict]], strategies: list[str], verbose: bool = False) -> list[dict]:
    shipments = rows_by_table["t_outbound_shipment"]
    demand_by_id: dict[str, list[float]] = defaultdict(list)
    for r in shipments:
        demand_by_id[r["product_id"]].append(_shipped_qty(r))

    unit_cost_by_id: dict[str, float] = {}
    product_group_by_id: dict[str, str] = {}
    for r in rows_by_table.get("t_product", []):
        pid = r.get("product_id")
        if pid is None:
            continue
        if r.get("unit_cost") is not None:
            unit_cost_by_id[pid] = float(r["unit_cost"])
        if r.get("product_group_id") is not None:
            product_group_by_id[pid] = r["product_group_id"]
    _vlog(verbose, f"{len(unit_cost_by_id)} item(s) have a usable unit_cost, {len(product_group_by_id)} have a product_group_id")

    # Hard rule: a revenue-based strategy must never silently drop or
    # price-fallback a shipped item -- catches BOTH a null unit_cost row
    # AND a product with no t_product row at all (the gap a
    # missing_cost_ids-only check used to miss).
    shipped_ids = set(demand_by_id)
    unpriced_ids = sorted(shipped_ids - set(unit_cost_by_id))
    if _REVENUE_STRATEGIES & set(strategies) and unpriced_ids:
        raise ValueError(
            f"{len(unpriced_ids)} shipped product(s) have no usable t_product.unit_cost "
            f"(missing row or null cost) -- e.g. {unpriced_ids[:5]}. Ask the user how to handle "
            "these before computing a revenue-based strategy; do not silently drop them or fall "
            "back to another price source."
        )

    revenue_periods_by_id: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in shipments:
        pid = r["product_id"]
        cost = unit_cost_by_id.get(pid)
        if cost is None:
            continue
        revenue_periods_by_id[pid][_period_key(r.get("actual_ship_date"))] += _shipped_qty(r) * cost
    revenue_by_id = {pid: sum(periods.values()) for pid, periods in revenue_periods_by_id.items()}
    revenue_series_by_id = {pid: list(periods.values()) for pid, periods in revenue_periods_by_id.items()}
    for pid, rev in sorted(revenue_by_id.items()):
        _vlog(verbose, f"revenue[{pid}] = {rev:.2f} (unit_cost={unit_cost_by_id[pid]}, {len(revenue_series_by_id[pid])} period(s))")

    if _REVENUE_STRATEGIES & set(strategies):
        # Belt-and-suspenders: the check above should make this impossible.
        # If a future refactor reintroduces a silent-drop path, fail loudly
        # here instead of quietly shipping an incomplete segmentation.
        assert shipped_ids <= set(revenue_by_id), (
            f"internal error: {sorted(shipped_ids - set(revenue_by_id))} were shipped but "
            "produced no revenue -- the unpriced-item check above should have caught this"
        )

    adi_cv2_by_id = {pid: adi_cv2_class(hist) for pid, hist in demand_by_id.items()} if "adi_cv2" in strategies else {}
    if adi_cv2_by_id:
        _vlog(verbose, f"adi_cv2_class: {adi_cv2_by_id}")
    abc_by_id = abc_bucket(revenue_by_id) if {"abc_revenue", "abc_xyz_matrix"} & set(strategies) else {}
    if abc_by_id:
        _vlog(verbose, f"abc_class: {abc_by_id}")
    xyz_by_id = xyz_bucket(revenue_series_by_id) if {"xyz_variability", "abc_xyz_matrix"} & set(strategies) else {}
    if xyz_by_id:
        _vlog(verbose, f"xyz_class: {xyz_by_id}")
    matrix_by_id = abc_xyz_matrix(abc_by_id, xyz_by_id) if "abc_xyz_matrix" in strategies else {}

    abc_hierarchy_by_id = {}
    if "abc_revenue_by_hierarchy" in strategies:
        # Roll each item up one level: its own product_group_id, or that
        # group's parent_product_group_id if t_product_hierarchy maps one --
        # an item with no group of its own is left as its own singleton
        # group rather than dropped.
        parent_by_group = {
            r["product_group_id"]: r["parent_product_group_id"]
            for r in rows_by_table.get("t_product_hierarchy", [])
            if r.get("product_group_id") is not None and r.get("parent_product_group_id")
        }

        def _rollup_key(pid: str) -> str:
            group = product_group_by_id.get(pid)
            return parent_by_group.get(group, group) if group else pid

        revenue_by_group: dict[str, float] = defaultdict(float)
        for pid, rev in revenue_by_id.items():
            revenue_by_group[_rollup_key(pid)] += rev
        _vlog(verbose, f"rolled-up group revenue: {dict(revenue_by_group)}")
        bucket_by_group = abc_bucket(revenue_by_group)
        _vlog(verbose, f"group -> bucket: {bucket_by_group}")
        abc_hierarchy_by_id = {pid: bucket_by_group[_rollup_key(pid)] for pid in revenue_by_id}

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    all_ids = sorted(set(demand_by_id) | set(revenue_by_id))
    return [
        {
            "run_timestamp": run_ts,
            "product_id": pid,
            "strategies_run": ",".join(strategies),
            "adi_cv2_class": adi_cv2_by_id.get(pid),
            "revenue": revenue_by_id.get(pid),
            "abc_class": abc_by_id.get(pid),
            "xyz_class": xyz_by_id.get(pid),
            "abc_xyz_matrix": matrix_by_id.get(pid),
            "abc_hierarchy_class": abc_hierarchy_by_id.get(pid),
        }
        for pid in all_ids
    ]


def run(args) -> tuple[list[dict], Path]:
    # Validate --customer before anything that costs real money/time
    # (a live Databricks query) or writes anywhere -- a bad customer name
    # must never let a warehouse query run first and fail last.
    _validate_customer(args.customer)
    errors = validate(args.strategies, args.tables)
    if errors:
        raise ValueError("; ".join(errors))

    verbose = getattr(args, "verbose", False)
    rows_by_table = fetch_rows(args.catalog, args.schema, args.tables, args.profile, verbose=verbose)
    records = compute_segments(rows_by_table, args.strategies, verbose=verbose)

    pull_sql = format_pull_sql_for_audit(build_pull_statements(args.catalog, args.schema, args.tables))
    create_sql, insert_sql = build_write_back_sql(args.catalog, args.schema, records)
    audit_sql = pull_sql + "\n\n" + create_sql + (("\n\n" + insert_sql) if insert_sql else "")
    audit_path = write_audit_file(audit_sql, args.customer, Path(args.out_dir) if args.out_dir else None)
    _vlog(verbose, f"audit SQL written -> {audit_path}")

    if not args.dry_run:
        _vlog(verbose, "executing CREATE TABLE against Databricks")
        run_query(create_sql, args.profile)
        if insert_sql:
            _vlog(verbose, "executing INSERT against Databricks")
            run_query(insert_sql, args.profile)

    return records, audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Never inferred -- ask the user which catalog")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--profile", required=True, help="Databricks CLI profile -- never guess/default this")
    parser.add_argument("--customer", required=True, help="Never inferred from --catalog -- used only for the local audit path")
    parser.add_argument("--strategies", nargs="+", required=True, help="e.g. adi_cv2 abc_revenue (registry: see segmentation_strategies.py)")
    parser.add_argument("--tables", nargs="+", default=DEFAULT_TABLES, help=f"Tables to pull; default {DEFAULT_TABLES}")
    parser.add_argument("--dry-run", action="store_true", help="Write the audit SQL file but skip executing the write-back against Databricks")
    parser.add_argument("--out-dir", default=None, help="Override the audit-file base dir (default experiments/segmentation_runs)")
    parser.add_argument("--verbose", action="store_true", help="Print each SQL query, row count, and intermediate per-item calculation to stderr as it happens")
    args = parser.parse_args()
    try:
        records, audit_path = run(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"computed {len(records)} record(s); audit SQL -> {audit_path}")
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
