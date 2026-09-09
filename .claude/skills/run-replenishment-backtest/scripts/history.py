#!/usr/bin/env python3
"""Print past backtest.py runs from experiments/results.jsonl -- check what's
already been tried before re-running an expensive sweep. See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOG_PATH = REPO_ROOT / "experiments" / "results.jsonl"


def _iter_rows(log_path: Path):
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for row in record.get("rows", []):
            yield {"timestamp": record.get("timestamp"), "data_source": record.get("data_source"), **row}


def _format_params(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items()) or "-"


def print_history(log_path: Path, strategy: str | None = None, limit: int | None = None) -> None:
    if not log_path.exists():
        print("No runs logged yet.")
        return
    rows = list(_iter_rows(log_path))
    if strategy:
        rows = [r for r in rows if r.get("strategy") == strategy]
    if not rows:
        print("No runs logged yet." if not strategy else f"No logged runs for strategy {strategy!r}.")
        return
    if limit:
        rows = rows[-limit:]
    print("| timestamp | data_source | strategy | params | trigger | fill_rate | total_cost |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        fill_rate = r.get("mean_fill_rate")
        total_cost = r.get("total_cost")
        fill_rate_str = f"{fill_rate:.4f}" if fill_rate is not None else "--"
        total_cost_str = f"{total_cost:,.2f}" if total_cost is not None else "--"
        tag = " (baseline)" if r.get("is_baseline") else ""
        print(f"| {r.get('timestamp')} | {r.get('data_source')} | {r.get('strategy')}{tag} | "
              f"{_format_params(r.get('params', {}))} | {r.get('trigger')} | {fill_rate_str} | {total_cost_str} |")


def diff_strategies(log_path: Path, strategy_a: str, strategy_b: str) -> None:
    """Compare the most recently logged run of two strategies -- cross-run
    diffing: what differs in config, and what differs in outcome."""
    if not log_path.exists():
        print("No runs logged yet.")
        return
    rows = list(_iter_rows(log_path))

    def latest_for(name: str):
        matches = [r for r in rows if r.get("strategy") == name]
        return matches[-1] if matches else None

    a, b = latest_for(strategy_a), latest_for(strategy_b)
    if a is None or b is None:
        missing = strategy_a if a is None else strategy_b
        print(f"No logged runs for strategy {missing!r}.")
        return

    def metric_row(label: str, key: str, fmt: str) -> str:
        va, vb = a.get(key), b.get(key)
        va_str = "--" if va is None else format(va, fmt)
        vb_str = "--" if vb is None else format(vb, fmt)
        delta_str = "--" if va is None or vb is None else format(vb - va, f"+{fmt}")
        return f"| {label} | {va_str} | {vb_str} | {delta_str} |"

    print(f"| | {strategy_a} | {strategy_b} | delta (B-A) |")
    print("|---|---|---|---|")
    print(f"| params | {_format_params(a.get('params', {}))} | {_format_params(b.get('params', {}))} | - |")
    print(f"| trigger | {a.get('trigger')} | {b.get('trigger')} | - |")
    print(f"| data_source | {a.get('data_source')} | {b.get('data_source')} | - |")
    print(f"| timestamp | {a.get('timestamp')} | {b.get('timestamp')} | - |")
    print(metric_row("fill_rate", "mean_fill_rate", ".4f"))
    print(metric_row("total_cost", "total_cost", ",.2f"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", default=None, help="Override the results log path (default: experiments/results.jsonl at repo root)")
    parser.add_argument("--strategy", default=None, help="Only show runs of this safety-stock strategy")
    parser.add_argument("--limit", type=int, default=None, help="Only show the N most recent matching runs")
    parser.add_argument("--diff", nargs=2, metavar=("STRATEGY_A", "STRATEGY_B"),
                         default=None, help="Compare the most recently logged run of two strategies")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    log_path = Path(args.log_path) if args.log_path else DEFAULT_LOG_PATH
    if args.diff:
        diff_strategies(log_path, args.diff[0], args.diff[1])
        return
    print_history(log_path, strategy=args.strategy, limit=args.limit)


if __name__ == "__main__":
    main()
