#!/usr/bin/env python3
"""Run a replenishment backtest for one strategy+trigger across every
unique_id in a dataset, and print an aggregate + per-item report.

This is the one place the per-unique_id grouping loop that every example
notebook under ../../../notebooks/ reinvents gets written once. See
../SKILL.md for the runbook and the repo root CLAUDE.md for the underlying
library pattern this script is a thin wrapper around.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from replenishment.io_ import (  # noqa: E402
    StandardSimulationRow,
    generate_standard_simulation_rows,
    load_standard_simulation_rows,
)
from replenishment.policy import ReplenishmentPolicy  # noqa: E402
from replenishment.simulation import simulate_replenishment  # noqa: E402
from replenishment.strategies.registry import ORDER_TRIGGERS, SAFETY_STOCK_STRATEGIES, describe  # noqa: E402
from replenishment.timeseries import TimeSeries  # noqa: E402


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _parse_params(raw: str | None) -> dict:
    if not raw:
        return {}
    params = {}
    for pair in raw.split(","):
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Malformed --params entry (expected key=value): {pair!r}")
        params[key.strip()] = _coerce(value.strip())
    return params


def _validate_params(strategy_name: str, params: dict) -> None:
    spec = describe(strategy_name)["params"]
    unknown = set(params) - set(spec)
    if unknown:
        raise ValueError(f"{strategy_name} has no param(s) {sorted(unknown)}. Valid params: {sorted(spec)}")
    missing = [name for name, meta in spec.items() if meta["required"] and name not in params]
    if missing:
        raise ValueError(f"{strategy_name} requires {missing}. Got: {sorted(params)}")


def _load_rows(args) -> list[StandardSimulationRow]:
    if args.data == "synthetic":
        if args.lead_time_field:
            raise ValueError("--lead-time-field is not supported with --data synthetic; use --lead-time <int> instead.")
        return generate_standard_simulation_rows(
            n_unique_ids=args.n_items, periods=args.periods,
            lead_time=args.lead_time if args.lead_time is not None else 1,
            seed=args.seed,
        )
    try:
        return load_standard_simulation_rows(
            args.data, actuals_field=args.actuals_field,
            initial_on_hand_field=args.initial_on_hand_field, lead_time_field=args.lead_time_field,
        )
    except ValueError as exc:
        if str(exc).startswith("Unsupported path"):
            raise ValueError(f"Unsupported --data {args.data!r} (use 'synthetic', a .csv, or a .parquet path)") from exc
        raise


@functools.lru_cache(maxsize=4)
def _read_raw_rows(path_str: str) -> tuple[dict, ...]:
    """Every row of --data as plain dicts (column name -> raw value),
    independent of the StandardSimulationRow schema -- used to resolve
    --moq-field/--review-period-field, which aren't part of that schema.
    Cached so requesting both in one run only reads the file once."""
    path = Path(path_str)
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            return tuple(csv.DictReader(fh))
    if path.suffix == ".parquet":
        import pandas as pd
        return tuple(pd.read_parquet(path).to_dict("records"))
    raise ValueError(f"Cannot read a raw column from {path_str!r} (use a .csv or .parquet path).")


def _load_column_by_unique_id(args, flag_name: str, column: str) -> dict[str, int]:
    if args.data == "synthetic":
        raise ValueError(f"{flag_name} is not supported with --data synthetic; use the matching scalar flag instead.")
    rows = _read_raw_rows(args.data)
    if not rows:
        raise ValueError("No rows loaded.")
    if column not in rows[0]:
        raise ValueError(f"{flag_name} column {column!r} not found in {args.data!r}; available columns: {sorted(rows[0])}")
    if "unique_id" not in rows[0]:
        raise ValueError(f"{args.data!r} has no 'unique_id' column, needed to resolve {flag_name}.")
    by_id: dict[str, int] = {}
    for row in rows:
        uid = str(row["unique_id"])
        if uid in by_id:
            continue
        try:
            by_id[uid] = int(row[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{flag_name} column {column!r} has a non-numeric value for unique_id {uid!r}: {row[column]!r}") from exc
    return by_id


def _resolve_field_overrides(args) -> tuple[dict | None, dict | None]:
    if args.moq is not None and args.moq_field:
        raise ValueError("--moq and --moq-field are mutually exclusive")
    if args.review_period is not None and args.review_period_field:
        raise ValueError("--review-period and --review-period-field are mutually exclusive")
    if args.lead_time is not None and args.lead_time_field:
        raise ValueError("--lead-time and --lead-time-field are mutually exclusive")
    moq_by_id = _load_column_by_unique_id(args, "--moq-field", args.moq_field) if args.moq_field else None
    review_period_by_id = (
        _load_column_by_unique_id(args, "--review-period-field", args.review_period_field)
        if args.review_period_field else None
    )
    return moq_by_id, review_period_by_id


def _group_by_unique_id(rows: list[StandardSimulationRow]) -> dict[str, list[StandardSimulationRow]]:
    grouped: dict[str, list[StandardSimulationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.unique_id].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: r.ds)
    return grouped


def _run_one_config(
    strategy: str, params: dict, trigger: str, grouped: dict, args,
    moq_by_id: dict | None = None, review_period_by_id: dict | None = None,
    forecast_field: str = "forecast",
) -> dict:
    """The per-unique_id loop, generalized to take an already-resolved
    (strategy, params, trigger) triple instead of reading args.strategy/
    args.params/args.trigger directly -- shared by the single-config and
    sweep paths so both run against identically-grouped rows.

    moq_by_id/review_period_by_id are dataset properties, resolved ONCE
    before any config runs (see _resolve_field_overrides) and shared by
    every config in a sweep -- unlike forecast_field, which is the thing
    actually being compared, so it's per-config."""
    strategy_cls = SAFETY_STOCK_STRATEGIES.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy {strategy!r}. Valid: {sorted(SAFETY_STOCK_STRATEGIES)}")
    trigger_cls = ORDER_TRIGGERS.get(trigger)
    if trigger_cls is None:
        raise ValueError(f"Unknown trigger {trigger!r}. Valid: {sorted(ORDER_TRIGGERS)}")
    _validate_params(strategy, params)

    if forecast_field != "forecast":
        sample_row = next(iter(grouped.values()))[0]
        if forecast_field not in sample_row.forecast_percentiles:
            raise ValueError(
                f"forecast_field {forecast_field!r} not found in forecast_percentiles; "
                f"available: {sorted(sample_row.forecast_percentiles)}"
            )
        if strategy != "NullSafetyStockStrategy":
            print(f"note: forecast_field {forecast_field!r} is a percentile forecast (already embeds its own "
                  f"buffer) paired with {strategy!r} (not NullSafetyStockStrategy) -- the buffer may be "
                  f"double-counted.")

    results = []
    errors = []
    for unique_id, group in grouped.items():
        try:
            if forecast_field == "forecast":
                forecast_values = [r.forecast for r in group]
            else:
                forecast_values = [r.forecast_percentiles[forecast_field] for r in group]
            forecast = TimeSeries.from_values(forecast_values)
            actuals = TimeSeries.from_values([r.actuals if r.actuals is not None else r.demand for r in group])
            lead_time = args.lead_time if args.lead_time is not None else group[0].lead_time
            if moq_by_id is not None:
                if unique_id not in moq_by_id:
                    raise ValueError(f"--moq-field has no value for unique_id {unique_id!r}")
                moq = moq_by_id[unique_id]
            else:
                moq = args.moq if args.moq is not None else 1
            if review_period_by_id is not None:
                if unique_id not in review_period_by_id:
                    raise ValueError(f"--review-period-field has no value for unique_id {unique_id!r}")
                review_period = review_period_by_id[unique_id]
            else:
                review_period = args.review_period if args.review_period is not None else 1
            policy = ReplenishmentPolicy(
                forecast=forecast, actuals=actuals,
                safety_stock=strategy_cls(**params), trigger=trigger_cls(),
                lead_time=lead_time, review_period=review_period,
                forecast_horizon=args.horizon, moq=moq,
            )
            result = simulate_replenishment(
                periods=len(group), demand=[r.demand for r in group],
                initial_on_hand=group[0].initial_on_hand, lead_time=lead_time, policy=policy,
                holding_cost_per_unit=args.holding_cost if args.holding_cost is not None else group[0].holding_cost_per_unit,
                stockout_cost_per_unit=args.stockout_cost if args.stockout_cost is not None else group[0].stockout_cost_per_unit,
                order_cost_per_order=args.order_cost if args.order_cost is not None else group[0].order_cost_per_order,
            )
            order_count = sum(1 for s in result.snapshots if s.order_placed > 0)
            results.append({"unique_id": unique_id, "summary": result.summary, "order_count": order_count})
        except Exception as exc:  # noqa: BLE001 -- report per-item, don't abort the batch
            errors.append((unique_id, str(exc)))

    return {
        "strategy": strategy, "params": params, "trigger": trigger, "forecast_field": forecast_field,
        "results": results, "errors": errors,
    }


def run_backtest(args) -> dict:
    """Single-config path -- unchanged return shape (results/errors keys)."""
    params = _parse_params(args.params)
    trigger = args.trigger or "OrderUpToTrigger"
    forecast_field = args.forecast_field or "forecast"
    moq_by_id, review_period_by_id = _resolve_field_overrides(args)
    rows = _load_rows(args)
    if not rows:
        raise ValueError("No rows loaded.")
    grouped = _group_by_unique_id(rows)
    outcome = _run_one_config(
        args.strategy, params, trigger, grouped, args,
        moq_by_id=moq_by_id, review_period_by_id=review_period_by_id, forecast_field=forecast_field,
    )
    outcome["is_baseline"] = False
    return outcome


def _load_configs(raw: str) -> list[dict]:
    path = Path(raw)
    text = path.read_text() if path.is_file() else raw
    try:
        configs = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--configs is not valid JSON (or an existing file path): {exc}") from exc
    if not isinstance(configs, list) or not configs:
        raise ValueError("--configs must be a non-empty JSON array of {strategy, params, trigger} objects")
    normalized = []
    for i, cfg in enumerate(configs):
        if "strategy" not in cfg:
            raise ValueError(f"--configs[{i}] is missing required key 'strategy'")
        normalized.append({
            "strategy": cfg["strategy"],
            "params": cfg.get("params", {}),
            "trigger": cfg.get("trigger", "OrderUpToTrigger"),
            "forecast_field": cfg.get("forecast_field", "forecast"),
        })
    return normalized


def run_sweep(args) -> list[dict]:
    """Load+group once, run every --configs entry against the same grouped
    rows, and (unless --no-baseline) prepend one NullSafetyStockStrategy
    baseline outcome per distinct trigger requested. moq/review_period are
    resolved once here (dataset properties, shared across every config);
    forecast_field is read per-config below (the thing being compared)."""
    configs = _load_configs(args.configs)
    moq_by_id, review_period_by_id = _resolve_field_overrides(args)
    rows = _load_rows(args)
    if not rows:
        raise ValueError("No rows loaded.")
    grouped = _group_by_unique_id(rows)

    variants = [
        _run_one_config(
            c["strategy"], c["params"], c["trigger"], grouped, args,
            moq_by_id=moq_by_id, review_period_by_id=review_period_by_id, forecast_field=c["forecast_field"],
        )
        for c in configs
    ]
    for outcome in variants:
        outcome["is_baseline"] = False

    baselines = []
    if not args.no_baseline and len(configs) >= 2:
        for trigger in sorted({c["trigger"] for c in configs}):
            baseline = _run_one_config(
                "NullSafetyStockStrategy", {}, trigger, grouped, args,
                moq_by_id=moq_by_id, review_period_by_id=review_period_by_id, forecast_field="forecast",
            )
            baseline["is_baseline"] = True
            baselines.append(baseline)

    return baselines + variants


def _aggregate(outcome: dict) -> dict:
    results = outcome["results"]
    if not results:
        return {
            "n_items": 0, "mean_fill_rate": None, "median_fill_rate": None, "total_cost": None,
            "holding_cost": None, "stockout_cost": None, "ordering_cost": None,
            "total_orders": None, "turnover": None,
        }
    fill_rates = [r["summary"].fill_rate for r in results]
    total_fulfilled = sum(r["summary"].total_fulfilled for r in results)
    total_on_hand = sum(r["summary"].average_on_hand for r in results)
    return {
        "n_items": len(results),
        "mean_fill_rate": statistics.mean(fill_rates),
        "median_fill_rate": statistics.median(fill_rates),
        "total_cost": sum(r["summary"].total_cost for r in results),
        "holding_cost": sum(r["summary"].holding_cost for r in results),
        "stockout_cost": sum(r["summary"].stockout_cost for r in results),
        "ordering_cost": sum(r["summary"].ordering_cost for r in results),
        "total_orders": sum(r["order_count"] for r in results),
        "turnover": (total_fulfilled / total_on_hand) if total_on_hand > 0 else None,
    }


def _print_report(outcome: dict) -> None:
    results = outcome["results"]
    errors = outcome["errors"]
    if outcome.get("forecast_field", "forecast") != "forecast":
        print(f"forecast_field: {outcome['forecast_field']}")
        print()
    if not results:
        print("No items simulated successfully.")
    else:
        agg = _aggregate(outcome)
        turnover_str = f"{agg['turnover']:.2f}" if agg["turnover"] is not None else "--"
        print("| items | mean fill_rate | median fill_rate | total_cost | holding_cost | stockout_cost | ordering_cost | turnover | orders |")
        print("|---|---|---|---|---|---|---|---|---|")
        print(f"| {agg['n_items']} | {agg['mean_fill_rate']:.4f} | {agg['median_fill_rate']:.4f} | {agg['total_cost']:,.2f} "
              f"| {agg['holding_cost']:,.2f} | {agg['stockout_cost']:,.2f} | {agg['ordering_cost']:,.2f} "
              f"| {turnover_str} | {agg['total_orders']} |")
        print()
        worst = sorted(results, key=lambda r: r["summary"].fill_rate)[:5]
        print("Worst 5 by fill_rate:")
        print("| unique_id | fill_rate | total_cost | avg_on_hand | turnover |")
        print("|---|---|---|---|---|")
        for r in worst:
            summary = r["summary"]
            item_turnover = summary.total_fulfilled / summary.average_on_hand if summary.average_on_hand > 0 else None
            item_turnover_str = f"{item_turnover:.2f}" if item_turnover is not None else "--"
            print(f"| {r['unique_id']} | {summary.fill_rate:.4f} | {summary.total_cost:,.2f} | {summary.avg_on_hand:.1f} | {item_turnover_str} |")
    if errors:
        print()
        print(f"{len(errors)} item(s) failed:")
        for unique_id, message in errors[:5]:
            print(f"- {unique_id}: {message}")
        if len(errors) > 5:
            print(f"...and {len(errors) - 5} more.")


def _format_params(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items()) or "-"


def _format_agg_row(agg: dict) -> str:
    if agg["n_items"] == 0:
        return "0 | -- | -- | --"
    return f"{agg['n_items']} | {agg['mean_fill_rate']:.4f} | {agg['median_fill_rate']:.4f} | {agg['total_cost']:,.2f}"


def _print_sweep_report(outcomes: list[dict]) -> None:
    baselines = {o["trigger"]: o for o in outcomes if o["is_baseline"]}
    variants = [o for o in outcomes if not o["is_baseline"]]

    if baselines:
        print("Baseline (NullSafetyStockStrategy -- zero safety-stock buffer):")
        print("| trigger | items | mean fill_rate | median fill_rate | total_cost |")
        print("|---|---|---|---|---|")
        for trigger in sorted(baselines):
            agg = _aggregate(baselines[trigger])
            print(f"| {trigger} | {_format_agg_row(agg)} |")
        print()

    ranked = sorted(variants, key=lambda o: (_aggregate(o)["total_cost"] is None, _aggregate(o)["total_cost"]))
    print("Variants (ranked by total_cost, ascending):")
    if baselines:
        print("| rank | strategy | params | trigger | forecast | items | fill_rate | vs baseline | total_cost | vs baseline |")
        print("|---|---|---|---|---|---|---|---|---|---|")
    else:
        print("| rank | strategy | params | trigger | forecast | items | fill_rate | total_cost |")
        print("|---|---|---|---|---|---|---|---|")
    for rank, outcome in enumerate(ranked, start=1):
        agg = _aggregate(outcome)
        params_str = _format_params(outcome["params"])
        forecast_field = outcome["forecast_field"]
        if not outcome["results"]:
            print(f"| {rank} | {outcome['strategy']} | {params_str} | {outcome['trigger']} | {forecast_field} | 0 | -- | -- |")
            continue
        base_outcome = baselines.get(outcome["trigger"])
        base_agg = _aggregate(base_outcome) if base_outcome else None
        if base_agg and base_agg["n_items"] > 0:
            fr_delta = agg["mean_fill_rate"] - base_agg["mean_fill_rate"]
            cost_delta = agg["total_cost"] - base_agg["total_cost"]
            print(f"| {rank} | {outcome['strategy']} | {params_str} | {outcome['trigger']} | {forecast_field} | {agg['n_items']} "
                  f"| {agg['mean_fill_rate']:.4f} | {fr_delta:+.4f} | {agg['total_cost']:,.2f} | {cost_delta:+,.2f} |")
        elif baselines:
            print(f"| {rank} | {outcome['strategy']} | {params_str} | {outcome['trigger']} | {forecast_field} | {agg['n_items']} "
                  f"| {agg['mean_fill_rate']:.4f} | n/a (baseline had 0 items) | {agg['total_cost']:,.2f} | n/a |")
        else:
            print(f"| {rank} | {outcome['strategy']} | {params_str} | {outcome['trigger']} | {forecast_field} | {agg['n_items']} "
                  f"| {agg['mean_fill_rate']:.4f} | {agg['total_cost']:,.2f} |")

    print()
    print("Cost breakdown:")
    print("| strategy | params | trigger | forecast | holding_cost | stockout_cost | ordering_cost | turnover | orders |")
    print("|---|---|---|---|---|---|---|---|---|")
    for outcome in outcomes:
        agg = _aggregate(outcome)
        label = f"{outcome['strategy']} (baseline)" if outcome["is_baseline"] else outcome["strategy"]
        if agg["n_items"] == 0:
            print(f"| {label} | {_format_params(outcome['params'])} | {outcome['trigger']} | {outcome['forecast_field']} | -- | -- | -- | -- | -- |")
            continue
        turnover_str = f"{agg['turnover']:.2f}" if agg["turnover"] is not None else "--"
        print(f"| {label} | {_format_params(outcome['params'])} | {outcome['trigger']} | {outcome['forecast_field']} "
              f"| {agg['holding_cost']:,.2f} | {agg['stockout_cost']:,.2f} | {agg['ordering_cost']:,.2f} "
              f"| {turnover_str} | {agg['total_orders']} |")

    for outcome in outcomes:
        if outcome["errors"]:
            print()
            print(f"{outcome['strategy']} ({outcome['trigger']}): {len(outcome['errors'])} item(s) failed, e.g. {outcome['errors'][0]}")


def _validate_customer(customer: str) -> None:
    if not customer or customer in (".", "..") or "/" in customer or "\\" in customer:
        raise ValueError(f"invalid --customer {customer!r} -- must be a plain name, no path separators")


def _resolve_log_path(args) -> Path:
    """experiments/results/<customer>/results.jsonl -- or experiments/results/_unscoped/
    when no --customer is given (e.g. a synthetic sanity-check run not tied to a real
    client). --log-path always wins over both. Shared by backtest.py, grid_search.py,
    and history.py so they can't drift on where a run's log actually lives."""
    if args.log_path:
        return Path(args.log_path)
    customer = getattr(args, "customer", None)
    if customer:
        _validate_customer(customer)
        return REPO_ROOT / "experiments" / "results" / customer / "results.jsonl"
    return REPO_ROOT / "experiments" / "results" / "_unscoped" / "results.jsonl"


def _iter_logged_rows(log_path: Path):
    if not log_path.exists():
        return
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for row in record.get("rows", []):
            yield {"timestamp": record.get("timestamp"), "data_source": record.get("data_source"), **row}


def _warn_on_repeat_configs(outcomes: list[dict], args) -> None:
    """Cross-run diffing, cheap version: flag when an identical
    (strategy, params, trigger, data_source) combo is already in the log,
    so a sweep doesn't silently re-run something already answered. Never
    blocks -- data or code may have changed since, so it's informational."""
    log_path = _resolve_log_path(args)
    prior_rows = list(_iter_logged_rows(log_path))
    if not prior_rows:
        return
    for outcome in outcomes:
        matches = [
            r for r in prior_rows
            if r["strategy"] == outcome["strategy"] and r["trigger"] == outcome["trigger"]
            and r["params"] == outcome["params"] and r["data_source"] == args.data
            and r.get("forecast_field", "forecast") == outcome["forecast_field"]
        ]
        if matches:
            latest = matches[-1]
            fill_rate = latest["mean_fill_rate"]
            total_cost = latest["total_cost"]
            fill_rate_str = f"{fill_rate:.4f}" if fill_rate is not None else "--"
            total_cost_str = f"{total_cost:,.2f}" if total_cost is not None else "--"
            print(f"note: {outcome['strategy']}({_format_params(outcome['params'])})/{outcome['trigger']} on "
                  f"{args.data!r} already run at {latest['timestamp']} -> "
                  f"fill_rate={fill_rate_str}, total_cost={total_cost_str}")


def _row_record(outcome: dict) -> dict:
    agg = _aggregate(outcome)
    return {
        "is_baseline": outcome["is_baseline"],
        "strategy": outcome["strategy"],
        "params": outcome["params"],
        "trigger": outcome["trigger"],
        "forecast_field": outcome["forecast_field"],
        "n_items": agg["n_items"],
        "n_errors": len(outcome["errors"]),
        "mean_fill_rate": agg["mean_fill_rate"],
        "median_fill_rate": agg["median_fill_rate"],
        "total_cost": agg["total_cost"],
        "holding_cost": agg["holding_cost"],
        "stockout_cost": agg["stockout_cost"],
        "ordering_cost": agg["ordering_cost"],
        "total_orders": agg["total_orders"],
        "turnover": agg["turnover"],
    }


def _log_results(outcomes: list[dict], args) -> None:
    if args.no_log:
        return
    log_path = _resolve_log_path(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": args.data,
        "rows": [_row_record(o) for o in outcomes],
    }
    with log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default=None, help="Safety-stock strategy class name, e.g. KRmseSafetyStock (required unless --configs is given)")
    parser.add_argument("--params", default=None, help="key=value,key2=value2 -- constructor kwargs for --strategy")
    parser.add_argument("--trigger", default=None, help="Order trigger class name (default: OrderUpToTrigger)")
    parser.add_argument("--configs", default=None, help="Path to a JSON file, or an inline JSON array, of {strategy, params, trigger} objects -- runs a sweep against the same dataset. Mutually exclusive with --strategy/--params/--trigger.")
    parser.add_argument("--no-baseline", action="store_true", help="Sweep mode only: skip the automatic NullSafetyStockStrategy baseline row")
    parser.add_argument("--customer", default=None, help="Partitions the results log at experiments/results/<customer>/results.jsonl; omit for experiments/results/_unscoped/results.jsonl. Never inferred -- ask if not given.")
    parser.add_argument("--log-path", default=None, help="Override the results log path entirely (default: experiments/results/<customer or _unscoped>/results.jsonl at repo root)")
    parser.add_argument("--no-log", action="store_true", help="Skip appending this run to the results log")
    parser.add_argument("--data", required=True, help="'synthetic', a .csv path, or a .parquet path")
    parser.add_argument("--forecast-field", default=None, help="Which forecast to use in single-config mode: 'forecast' (default) or a forecast_percentiles key, e.g. 'p90'. In --configs mode, set this per-entry instead.")

    lead_group = parser.add_mutually_exclusive_group()
    lead_group.add_argument("--lead-time", type=int, default=None, help="Override every item's lead_time with one value (else uses the data's own lead_time column)")
    lead_group.add_argument("--lead-time-field", default=None, help="Column to read lead_time from, if not the data's default 'lead_time' column")

    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument("--review-period", type=int, default=None, help="Uniform review period for every item (default: 1)")
    review_group.add_argument("--review-period-field", default=None, help="Column in --data holding each item's review period (dataset property, resolved once, shared by every --configs entry)")

    moq_group = parser.add_mutually_exclusive_group()
    moq_group.add_argument("--moq", type=int, default=None, help="Uniform MOQ for every item (default: 1)")
    moq_group.add_argument("--moq-field", default=None, help="Column in --data holding each item's MOQ (dataset property, resolved once, shared by every --configs entry)")

    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--holding-cost", type=float, default=None)
    parser.add_argument("--stockout-cost", type=float, default=None)
    parser.add_argument("--order-cost", type=float, default=None)
    parser.add_argument("--actuals-field", default=None, help="Column to use as actuals when loading CSV/parquet (default: 'actuals')")
    parser.add_argument("--initial-on-hand-field", default=None, help="Column to use as initial_on_hand when loading CSV/parquet")
    parser.add_argument("--n-items", type=int, default=5, help="--data synthetic only")
    parser.add_argument("--periods", type=int, default=90, help="--data synthetic only")
    parser.add_argument("--seed", type=int, default=7, help="--data synthetic only")
    return parser


def _validate_mode(args) -> None:
    singular_given = args.strategy is not None or args.params is not None or args.trigger is not None
    if args.configs and singular_given:
        raise ValueError("--configs is mutually exclusive with --strategy/--params/--trigger")
    if not args.configs and not args.strategy:
        raise ValueError("--strategy is required unless --configs is given")


def main() -> None:
    args = _build_arg_parser().parse_args()
    try:
        # Validate --customer before running anything -- a bad customer name
        # must never let a full backtest/sweep run first and fail only when
        # it tries to log, wasting the run and leaving nothing logged for it.
        if args.customer:
            _validate_customer(args.customer)
        _validate_mode(args)
        if args.configs:
            outcomes = run_sweep(args)
            _print_sweep_report(outcomes)
        else:
            outcomes = [run_backtest(args)]
            _print_report(outcomes[0])
        _warn_on_repeat_configs(outcomes, args)
        _log_results(outcomes, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
