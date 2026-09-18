#!/usr/bin/env python3
"""Persist a hypothesis + its validated strategy space as an auditable YAML
record -- one file per run, nested by customer (many customers, many runs;
see ../SKILL.md). Re-validates before writing: an invalid strategy space
(missing selection_rationale, too many candidates, unknown strategy/param)
can never be logged as if it were audited.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from validate_strategy_space import validate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "strategy_space_decisions"


def _slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "-".join(words) or "hypothesis"


def _validate_customer(customer: str) -> None:
    """--customer becomes a directory name under out_dir -- reject anything
    that could escape it (a path separator or '..') instead of silently
    writing the audit record somewhere else on disk."""
    if not customer or customer in (".", "..") or "/" in customer or "\\" in customer:
        raise ValueError(f"--customer {customer!r} must be a plain name -- no '/', '\\\\', or '..'")


def build_record(strategy_space: dict, hypothesis: str, customer: str) -> dict:
    return {
        "customer": customer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": hypothesis,
        "data_summary": strategy_space.get("data_summary", {}),
        "selection_rationale": strategy_space["selection_rationale"],
        "candidates": strategy_space["candidates"],
    }


def log_decision(strategy_space: dict, hypothesis: str, customer: str, out_dir: Path) -> Path:
    _validate_customer(customer)
    errors = validate(strategy_space)
    if errors:
        raise ValueError("refusing to log an invalid strategy space: " + "; ".join(errors))

    record = build_record(strategy_space, hypothesis, customer)
    customer_dir = out_dir / customer
    customer_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{timestamp_slug}-{_slugify(hypothesis)}.yaml"
    out_path = customer_dir / filename
    with out_path.open("w") as fh:
        yaml.safe_dump(record, fh, sort_keys=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-space", required=True, help="Path to a propose-strategy-space JSON output")
    parser.add_argument("--hypothesis", required=True, help="The original natural-language hypothesis")
    parser.add_argument("--customer", required=True, help="Customer/client identifier -- never inferred, always stated")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Root dir for the per-customer audit log")
    args = parser.parse_args()

    strategy_space = json.loads(Path(args.strategy_space).read_text())
    try:
        out_path = log_decision(strategy_space, args.hypothesis, args.customer, Path(args.out_dir))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"logged decision -> {out_path}")


if __name__ == "__main__":
    main()
