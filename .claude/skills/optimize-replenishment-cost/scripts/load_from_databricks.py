#!/usr/bin/env python3
"""Materialize a Databricks table as a local StandardSimulationRow-shaped
CSV, so grid_search.py / backtest.py's existing CSV loader can read it
unchanged. Shells out to `databricks experimental aitools tools query` (the
same CLI path databricks-core standardizes on) instead of adding a
Spark/SQL connector dependency. See ../SKILL.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys


def run_query(sql: str, profile: str) -> list[dict]:
    result = subprocess.run(
        ["databricks", "experimental", "aitools", "tools", "query", sql, "--profile", profile],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"databricks query failed:\n{result.stderr}")
    stdout = result.stdout.strip()
    if not stdout or not stdout.startswith(("[", "{")):
        # DDL/DML with nothing to return (e.g. CREATE TABLE) prints a plain
        # success message instead of JSON -- confirmed live: "Query executed
        # successfully (no results)". No rows to normalize either way.
        return []
    rows = json.loads(stdout)
    # The CLI's JSON output can't distinguish SQL NULL from an empty string --
    # confirmed live: a column with `IS NULL = true` still serializes as ""
    # here. That distinction is already lost before this function sees it, so
    # normalize "" -> None so every caller's `is not None` check behaves the
    # same as it would against a real NULL, instead of silently treating a
    # missing value as present.
    return [{k: (None if v == "" else v) for k, v in row.items()} for row in rows]


def _validate_table(table: str) -> None:
    # Shape check, not a whitelist (this loader is generic over any client
    # table, unlike segment-inventory-items' fixed small set) -- catches a
    # stray SQL fragment or typo'd --table before it becomes half a query.
    # Hyphens are valid in real catalog/schema names, so this doesn't
    # restrict to \w+.
    parts = table.split(".")
    if len(parts) != 3 or any(not p for p in parts):
        raise ValueError(f"invalid --table {table!r} -- must be catalog.schema.table (3 dot-separated parts)")


def write_table_to_csv(table: str, profile: str, out: str, where: str | None = None) -> int:
    _validate_table(table)
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    rows = run_query(sql, profile)
    if not rows:
        raise ValueError(f"{table!r} returned no rows.")

    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, help="catalog.schema.table")
    parser.add_argument("--profile", required=True, help="Databricks CLI profile -- never guess/default this, ask the user which one")
    parser.add_argument("--out", required=True, help="Local CSV path to write")
    parser.add_argument("--where", default=None, help="Optional SQL WHERE clause (no leading WHERE)")
    args = parser.parse_args()
    try:
        n = write_table_to_csv(args.table, args.profile, args.out, args.where)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
