#!/usr/bin/env python3
"""Hard-rule validator: every requested segmentation strategy's required
tables must be in --tables, or validation fails outright -- no path around
it. Mirrors propose-strategy-space/validate_strategy_space.py's hard rules.
See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from segmentation_strategies import SEGMENTATION_STRATEGIES  # noqa: E402


def validate(strategies: list[str], tables_supplied: list[str]) -> list[str]:
    errors = []
    supplied = set(tables_supplied)
    for name in strategies:
        if name not in SEGMENTATION_STRATEGIES:
            errors.append(f"unknown strategy {name!r} -- not in the registry ({sorted(SEGMENTATION_STRATEGIES)})")
            continue
        required = set(SEGMENTATION_STRATEGIES[name]["required_tables"])
        missing = required - supplied
        if missing:
            errors.append(
                f"strategy {name!r} requires {sorted(required)}, but {sorted(missing)} "
                "was not supplied -- pull it explicitly with --tables before running this strategy"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", required=True, help="Strategy names to validate, e.g. adi_cv2 abc_revenue")
    parser.add_argument("--tables", nargs="+", required=True, help="Tables actually supplied/queried this run, e.g. t_outbound_shipment t_product")
    args = parser.parse_args()
    errors = validate(args.strategies, args.tables)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
    print("ok")


if __name__ == "__main__":
    main()
