#!/usr/bin/env python3
"""Build the exact SQL this skill runs (pulls + append-only write-back) and
log it as an auditable file before/alongside execution --
experiments/segmentation_runs/<customer>/<timestamp>.sql. Mirrors
propose-strategy-space/log_decision.py's per-customer per-run audit trail.
See ../SKILL.md.

Write-back is append-only by construction: CREATE TABLE IF NOT EXISTS +
INSERT are the only statements this module can produce. There is no
DROP/MERGE/DELETE/overwrite path here -- run_segmentation.py has nothing
else to execute.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

# ponytail: raw f-string SQL, no bind params -- the Databricks CLI query
# path (run_query in optimize-replenishment-cost/load_from_databricks.py)
# already takes a single SQL string everywhere else in this repo. Escape
# string literals below; there's no external/untrusted input on this path.
_TABLE_SELECT_COLUMNS = {
    "t_outbound_shipment": ["product_id", "actual_ship_date", "shipped_qty"],
    "t_product": ["id AS product_id", "unit_cost", "product_group_id"],
    "t_product_hierarchy": ["id AS product_group_id", "parent_product_group_id"],
}

_SEGMENTATION_TABLE_COLUMNS = [
    "run_timestamp", "product_id", "strategies_run",
    "adi_cv2_class", "revenue", "abc_class", "xyz_class", "abc_xyz_matrix",
    "abc_hierarchy_class",
]


def _validate_customer(customer: str) -> None:
    if not customer or customer in (".", "..") or "/" in customer or "\\" in customer:
        raise ValueError(f"invalid --customer {customer!r} -- must be a plain name, no path separators")


def build_pull_statements(catalog: str, schema: str, tables: list[str]) -> dict[str, str]:
    if "t_outbound_shipment" not in tables:
        raise ValueError("t_outbound_shipment is always required -- it's the base demand/revenue signal")
    statements = {}
    for table in tables:
        if table not in _TABLE_SELECT_COLUMNS:
            raise ValueError(f"unknown table {table!r} -- add its column list to _TABLE_SELECT_COLUMNS first")
        cols = ", ".join(_TABLE_SELECT_COLUMNS[table])
        statements[table] = f"SELECT {cols} FROM {catalog}.{schema}.{table}"
    return statements


def format_pull_sql_for_audit(statements: dict[str, str]) -> str:
    return "\n\n".join(f"-- pull: {table}\n{sql};" for table, sql in statements.items())


def _sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def build_write_back_sql(catalog: str, schema: str, records: list[dict]) -> tuple[str, str | None]:
    """Returns (create_table_sql, insert_sql_or_None). Append-only:
    CREATE TABLE IF NOT EXISTS + INSERT are the only statements produced."""
    table = f"{catalog}.{schema}.t_segmentation"
    columns_sql = ",\n  ".join(
        f"{c} TIMESTAMP" if c == "run_timestamp" else f"{c} STRING" if c != "revenue" else f"{c} DOUBLE"
        for c in _SEGMENTATION_TABLE_COLUMNS
    )
    create_sql = f"CREATE TABLE IF NOT EXISTS {table} (\n  {columns_sql}\n);"
    if not records:
        return create_sql, None
    rows_sql = []
    for r in records:
        vals = ", ".join(_sql_literal(r.get(c)) for c in _SEGMENTATION_TABLE_COLUMNS)
        rows_sql.append(f"({vals})")
    insert_sql = (
        f"INSERT INTO {table} ({', '.join(_SEGMENTATION_TABLE_COLUMNS)}) VALUES\n"
        + ",\n".join(rows_sql) + ";"
    )
    return create_sql, insert_sql


def write_audit_file(sql_text: str, customer: str, out_dir: Path | None = None) -> Path:
    _validate_customer(customer)
    base = out_dir or (REPO_ROOT / "experiments" / "segmentation_runs")
    target_dir = base / customer
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = target_dir / f"{ts}.sql"
    path.write_text(sql_text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--customer", required=True, help="Never inferred from --catalog -- ask if not given")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    try:
        statements = build_pull_statements(args.catalog, args.schema, args.tables)
        out_dir = Path(args.out_dir) if args.out_dir else None
        path = write_audit_file(format_pull_sql_for_audit(statements), args.customer, out_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
