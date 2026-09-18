#!/usr/bin/env python3
"""Classify each unique_id's demand shape (Syntetos-Boylan) from a
StandardSimulationRow-shaped CSV/parquet, so propose-strategy-space can
route strategy candidates by what the data actually looks like, not just
what the hypothesis claims. See ../SKILL.md.

Loads via replenishment.io_.load_standard_simulation_rows -- the same
function run-replenishment-backtest's backtest.py uses -- so
--actuals-field/--initial-on-hand-field/--lead-time-field behave identically
in both skills and can't drift out of sync with each other.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from replenishment.classification import classify_demand  # noqa: E402
from replenishment.io_ import load_standard_simulation_rows  # noqa: E402


def load_rows(path: str, *, actuals_field: str | None = None, initial_on_hand_field: str | None = None,
              lead_time_field: str | None = None):
    return load_standard_simulation_rows(
        path, actuals_field=actuals_field,
        initial_on_hand_field=initial_on_hand_field, lead_time_field=lead_time_field,
    )


def classify(rows) -> dict:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.unique_id].append(r)

    out = {}
    for unique_id, group in grouped.items():
        group.sort(key=lambda r: r.ds)
        history = [r.actuals if r.actuals is not None else r.demand for r in group]
        history = [h or 0 for h in history]
        out[unique_id] = {
            "demand_class": classify_demand(history),
            "n_periods": len(history),
            "has_actuals": any(r.actuals is not None for r in group),
            "mean_demand": (sum(history) / len(history)) if history else 0.0,
            "zero_share": (sum(1 for h in history if h == 0) / len(history)) if history else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="A .csv or .parquet path (StandardSimulationRow shape)")
    parser.add_argument("--actuals-field", default=None, help="Column to use as actuals, if not the data's default 'actuals' column")
    parser.add_argument("--initial-on-hand-field", default=None, help="Column to use as initial_on_hand, if not the data's default column")
    parser.add_argument("--lead-time-field", default=None, help="Column to use as lead_time, if not the data's default 'lead_time' column")
    args = parser.parse_args()
    try:
        rows = load_rows(
            args.data, actuals_field=args.actuals_field,
            initial_on_hand_field=args.initial_on_hand_field, lead_time_field=args.lead_time_field,
        )
        if not rows:
            raise ValueError("No rows loaded.")
        result = classify(rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
