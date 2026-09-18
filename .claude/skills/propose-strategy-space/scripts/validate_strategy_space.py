#!/usr/bin/env python3
"""Validate a propose-strategy-space output against the real strategy/trigger
registry -- catches a hallucinated strategy/trigger/param name or a missing
required param before it reaches optimize-replenishment-cost's grid
expansion. Reuses registry.describe() instead of re-encoding constructor
signatures.

Also enforces two hard audit rules (not just registry validity): a strategy
space may name at most half of the available safety-stock strategies (a
hypothesis must narrow to a real subset, never sweep everything), and must
carry a top-level `selection_rationale` grounded in the data's actual
`demand_classes_present` -- so "why K strategies, not all of them" is always
answerable from the file itself. See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from replenishment.strategies.registry import (  # noqa: E402
    describe,
    list_order_triggers,
    list_safety_stock_strategies,
)


def _validate_selection_rationale(space: dict) -> list[str]:
    """Hard rule: a strategy space must carry a top-level `selection_rationale`
    explaining why this subset (not every available strategy) was chosen,
    and it must actually be grounded in the data -- referencing at least one
    of `data_summary.demand_classes_present` -- not filler text."""
    rationale = space.get("selection_rationale")
    if not rationale or not isinstance(rationale, str) or not rationale.strip():
        return ["strategy space is missing a non-empty top-level 'selection_rationale'"]
    demand_classes = space.get("data_summary", {}).get("demand_classes_present", [])
    if demand_classes and not any(dc.lower() in rationale.lower() for dc in demand_classes):
        return [
            f"'selection_rationale' doesn't reference any of data_summary.demand_classes_present "
            f"{demand_classes} -- it must be grounded in the actual data, not generic text"
        ]
    return []


def validate(space: dict) -> list[str]:
    errors: list[str] = []
    candidates = space.get("candidates")
    if not candidates:
        return ["strategy space has no non-empty 'candidates' list"]

    max_candidates = len(list_safety_stock_strategies()) // 2
    if len(candidates) > max_candidates:
        errors.append(
            f"strategy space has {len(candidates)} candidates, but at most {max_candidates} "
            f"(half of the {len(list_safety_stock_strategies())} available strategies) are allowed -- "
            "a hypothesis must narrow to a real subset, not sweep everything"
        )

    errors.extend(_validate_selection_rationale(space))

    strategies = set(list_safety_stock_strategies())
    triggers = set(list_order_triggers())

    for i, c in enumerate(candidates):
        strategy = c.get("strategy")
        trigger = c.get("trigger", "OrderUpToTrigger")
        if strategy not in strategies:
            errors.append(f"candidates[{i}]: unknown strategy {strategy!r}. Valid: {sorted(strategies)}")
            continue
        if trigger not in triggers:
            errors.append(f"candidates[{i}] ({strategy}): unknown trigger {trigger!r}. Valid: {sorted(triggers)}")

        spec = describe(strategy)["params"]

        # A param typed `object` (e.g. DemandBufferDecorator.wrapped, which
        # needs an already-constructed SafetyStockStrategy instance) can't
        # be expressed in a JSON param_grid at all -- reject the candidate
        # outright instead of letting a string sail through validation and
        # crash per-item at simulation time with a buried error.
        object_typed = sorted(name for name, meta in spec.items() if meta["type"] and "object" in meta["type"])
        if object_typed:
            errors.append(
                f"candidates[{i}] ({strategy}): requires strategy-object param(s) {object_typed}, which a "
                "JSON param_grid can't express -- not usable as a top-level candidate through this pipeline"
            )
            continue

        grid = c.get("param_grid", {})
        unknown = set(grid) - set(spec)
        if unknown:
            errors.append(f"candidates[{i}] ({strategy}): unknown param(s) {sorted(unknown)}. Valid: {sorted(spec)}")
        missing = [name for name, meta in spec.items() if meta["required"] and name not in grid]
        if missing:
            errors.append(f"candidates[{i}] ({strategy}): missing required param(s) {missing}")
        for name, values in grid.items():
            if not isinstance(values, list) or not values:
                errors.append(f"candidates[{i}] ({strategy}).param_grid[{name!r}] must be a non-empty list")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-space", required=True)
    args = parser.parse_args()
    space = json.loads(Path(args.strategy_space).read_text())
    errors = validate(space)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
