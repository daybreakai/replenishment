#!/usr/bin/env python3
"""Deterministic eval for propose-strategy-space output. For a fixed
hypothesis fixture (hypotheses.jsonl), checks that a produced
strategy_space.json is registry-valid and includes/excludes the right
strategy families -- no LLM grading, see ../SKILL.md for why.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from validate_strategy_space import validate  # noqa: E402


def load_hypotheses(path: str) -> dict[str, dict]:
    fixtures = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            fixture = json.loads(line)
            fixtures[fixture["id"]] = fixture
    return fixtures


def grade(fixture: dict, space: dict) -> list[str]:
    errors = validate(space)
    if errors:
        return errors
    strategies = {c["strategy"] for c in space.get("candidates", [])}
    must_include_any = set(fixture.get("must_include_any", []))
    must_not_include = set(fixture.get("must_not_include", []))
    if must_include_any and not (strategies & must_include_any):
        errors.append(f"expected at least one of {sorted(must_include_any)}, got {sorted(strategies)}")
    disallowed = strategies & must_not_include
    if disallowed:
        errors.append(f"got disallowed strategy(ies) {sorted(disallowed)} for this hypothesis")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", default=str(Path(__file__).parent / "hypotheses.jsonl"))
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--strategy-space", required=True)
    args = parser.parse_args()

    fixtures = load_hypotheses(args.hypotheses)
    if args.hypothesis_id not in fixtures:
        raise SystemExit(f"error: unknown --hypothesis-id {args.hypothesis_id!r}. Valid: {sorted(fixtures)}")

    space = json.loads(Path(args.strategy_space).read_text())
    errors = grade(fixtures[args.hypothesis_id], space)
    print(json.dumps({"passed": not errors, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
