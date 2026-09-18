#!/usr/bin/env python3
"""Expand a propose-strategy-space JSON's param_grids into concrete configs,
run them all through run-replenishment-backtest's existing sweep engine, and
report the min-total_cost config(s) that still clear the given constraints.

No new simulation engine -- this is a grid-expansion + constraint-filter
layer reusing backtest.py's _run_one_config/_aggregate/_load_rows directly
(reaching into its private helpers deliberately, instead of duplicating the
per-unique_id loop a second time). See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / ".claude/skills/run-replenishment-backtest/scripts"))
sys.path.insert(0, str(REPO_ROOT / ".claude/skills/propose-strategy-space/scripts"))

import backtest  # noqa: E402
from validate_strategy_space import validate as validate_strategy_space  # noqa: E402


def expand_candidates(strategy_space: dict) -> list[dict]:
    """One candidate's param_grid -> N configs (cartesian product of its
    values), each shaped like a backtest.py --configs entry."""
    configs = []
    for candidate in strategy_space.get("candidates", []):
        strategy = candidate["strategy"]
        trigger = candidate.get("trigger", "OrderUpToTrigger")
        forecast_field = candidate.get("forecast_field", "forecast")
        grid = candidate.get("param_grid", {})
        if not grid:
            configs.append({"strategy": strategy, "params": {}, "trigger": trigger, "forecast_field": forecast_field})
            continue
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            configs.append({
                "strategy": strategy, "params": dict(zip(keys, values)),
                "trigger": trigger, "forecast_field": forecast_field,
            })
    return configs


def _passes_constraints(outcome: dict, args) -> bool:
    agg = backtest._aggregate(outcome)
    if agg["n_items"] == 0:
        return False
    if args.min_fill_rate is not None and agg["mean_fill_rate"] < args.min_fill_rate:
        return False
    if args.max_total_cost is not None and agg["total_cost"] > args.max_total_cost:
        return False
    return True


def _vlog(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[verbose] {msg}", file=sys.stderr)


def run_grid_search(args) -> dict:
    verbose = getattr(args, "verbose", False)
    strategy_space = json.loads(Path(args.strategy_space).read_text())
    errors = validate_strategy_space(strategy_space)
    if errors:
        raise ValueError("strategy space failed validation: " + "; ".join(errors))
    _vlog(verbose, f"strategy space validated: {len(strategy_space.get('candidates', []))} candidate(s)")

    configs = expand_candidates(strategy_space)
    if not configs:
        raise ValueError("strategy space produced zero configs -- check 'candidates'.")
    _vlog(verbose, f"expanded to {len(configs)} config(s): {[(c['strategy'], c['params']) for c in configs]}")

    moq_by_id, review_period_by_id = backtest._resolve_field_overrides(args)
    rows = backtest._load_rows(args)
    if not rows:
        raise ValueError("No rows loaded.")
    grouped = backtest._group_by_unique_id(rows)
    _vlog(verbose, f"loaded {len(rows)} row(s) across {len(grouped)} item(s)")

    outcomes = []
    for i, c in enumerate(configs, start=1):
        _vlog(verbose, f"running config {i}/{len(configs)}: {c['strategy']}({backtest._format_params(c['params'])})/{c['trigger']}")
        outcome = backtest._run_one_config(
            c["strategy"], c["params"], c["trigger"], grouped, args,
            moq_by_id=moq_by_id, review_period_by_id=review_period_by_id, forecast_field=c["forecast_field"],
        )
        agg = backtest._aggregate(outcome)
        _vlog(verbose, f"  -> fill_rate={agg['mean_fill_rate']}, total_cost={agg['total_cost']}, errors={len(outcome['errors'])}")
        outcomes.append(outcome)
    for outcome in outcomes:
        outcome["is_baseline"] = False

    survivors = [o for o in outcomes if _passes_constraints(o, args)]
    survivors.sort(key=lambda o: backtest._aggregate(o)["total_cost"])
    _vlog(verbose, f"{len(survivors)}/{len(outcomes)} config(s) survived constraints")
    return {"outcomes": outcomes, "survivors": survivors, "n_configs": len(configs)}


def _print_errors(result: dict) -> None:
    """Surface which configs had items error out -- without this, '0
    survivors' is indistinguishable from 'every item crashed' vs. 'ran fine
    but missed the constraint', same gap backtest.py's own sweep report
    already closes for single/sweep runs."""
    errored = [o for o in result["outcomes"] if o["errors"]]
    if not errored:
        return
    print()
    print(f"{len(errored)} config(s) had item-level errors:")
    for outcome in errored:
        params_str = backtest._format_params(outcome["params"])
        print(f"- {outcome['strategy']}({params_str})/{outcome['trigger']}: "
              f"{len(outcome['errors'])} item(s) failed, e.g. {outcome['errors'][0]}")


def _print_report(result: dict, args) -> None:
    survivors = result["survivors"][: args.top]
    print(f"{result['n_configs']} config(s) tried, {len(result['survivors'])} survived constraints.")
    _print_errors(result)
    if not survivors:
        print("No config satisfied the constraints -- loosen --min-fill-rate/--max-total-cost, "
              "or widen the strategy space's param_grid values.")
        return
    print()
    print("| rank | strategy | params | trigger | forecast | fill_rate | total_cost | holding | stockout | ordering |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for rank, outcome in enumerate(survivors, start=1):
        agg = backtest._aggregate(outcome)
        params_str = backtest._format_params(outcome["params"])
        print(f"| {rank} | {outcome['strategy']} | {params_str} | {outcome['trigger']} | {outcome['forecast_field']} "
              f"| {agg['mean_fill_rate']:.4f} | {agg['total_cost']:,.2f} | {agg['holding_cost']:,.2f} "
              f"| {agg['stockout_cost']:,.2f} | {agg['ordering_cost']:,.2f} |")
    best = survivors[0]
    print()
    print(f"Best under constraints: {best['strategy']}({backtest._format_params(best['params'])}) / {best['trigger']}")


def _log_results(result: dict, args) -> None:
    if args.no_log:
        return
    log_path = backtest._resolve_log_path(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": args.data,
        "source": "grid_search",
        "rows": [backtest._row_record(o) for o in result["outcomes"]],
    }
    with log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


_DEAD_BACKTEST_FLAGS = {"strategy", "params", "trigger", "configs", "no_baseline", "forecast_field"}


def _strip_dead_flags(parser: argparse.ArgumentParser) -> None:
    """backtest._build_arg_parser() defines --strategy/--params/--trigger/
    --configs/--no-baseline/--forecast-field for its own single-vs-sweep
    mode switch. grid_search.py never reads any of them -- trigger and
    forecast_field always come from the strategy space's own per-candidate
    fields -- so leaving them in would make --help advertise flags that
    silently no-op if passed. Remove them from the parser entirely."""
    for action in list(parser._actions):
        if action.dest in _DEAD_BACKTEST_FLAGS:
            parser._remove_action(action)
            for opt in action.option_strings:
                parser._option_string_actions.pop(opt, None)
            # _remove_action only touches parser._actions; the help formatter
            # renders from each _action_groups' own _group_actions list, so
            # without this the flag vanishes from usage but not from --help.
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = backtest._build_arg_parser()
    _strip_dead_flags(parser)
    parser.description = __doc__
    parser.add_argument("--strategy-space", required=True, help="Path to a propose-strategy-space JSON output")
    parser.add_argument("--min-fill-rate", type=float, default=None, help="Constraint: minimum mean fill_rate to survive")
    parser.add_argument("--max-total-cost", type=float, default=None, help="Constraint: maximum total_cost to survive")
    parser.add_argument("--verbose", action="store_true", help="Print each config's params and result to stderr as it runs, not just the final table")
    parser.add_argument("--top", type=int, default=5, help="How many survivors to print in the ranked table")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    try:
        if args.customer:
            backtest._validate_customer(args.customer)
        result = run_grid_search(args)
        _print_report(result, args)
        _log_results(result, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
