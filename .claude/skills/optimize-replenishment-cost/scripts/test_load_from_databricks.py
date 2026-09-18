"""ponytail self-check for load_from_databricks.py -- run
`python test_load_from_databricks.py`. Fakes run_query directly (no real
databricks CLI call) since this is a pure CSV-shaping test."""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import load_from_databricks as mod  # noqa: E402


def test_run_query_normalizes_empty_string_to_none():
    # Discovered against a real Databricks table: the CLI's JSON output
    # serializes SQL NULL as "" (confirmed live via `IS NULL` = true on a
    # "" value), not as JSON null. Every caller checking `is not None` for
    # a missing value must see a real None, not "".
    original = mod.subprocess.run

    def fake_run(cmd, capture_output, text):
        return type("R", (), {
            "returncode": 0,
            "stdout": '[{"id": "A", "unit_cost": ""}, {"id": "B", "unit_cost": "5.0"}]',
            "stderr": "",
        })()
    mod.subprocess.run = fake_run
    try:
        rows = mod.run_query("SELECT 1", "p")
        assert rows == [{"id": "A", "unit_cost": None}, {"id": "B", "unit_cost": "5.0"}], rows
    finally:
        mod.subprocess.run = original


def test_run_query_handles_non_json_ddl_success_message():
    # Discovered live: a CREATE TABLE / DDL statement returns the plain
    # text "Query executed successfully (no results)", not JSON -- must not
    # crash json.loads, and there are no rows to normalize either way.
    original = mod.subprocess.run

    def fake_run(cmd, capture_output, text):
        return type("R", (), {"returncode": 0, "stdout": "Query executed successfully (no results)", "stderr": ""})()
    mod.subprocess.run = fake_run
    try:
        assert mod.run_query("CREATE TABLE x (y INT)", "p") == []
    finally:
        mod.subprocess.run = original


def test_writes_csv_from_query_rows():
    rows = [{"unique_id": "A", "demand": "10"}, {"unique_id": "B", "demand": "5"}]
    original = mod.run_query
    mod.run_query = lambda sql, profile: rows
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.csv"
            n = mod.write_table_to_csv("cat.schema.tbl", "p", str(out))
            assert n == 2
            with out.open() as fh:
                written = list(csv.DictReader(fh))
            assert written == rows, written
    finally:
        mod.run_query = original


def test_where_clause_is_appended_to_sql():
    captured = {}
    original = mod.run_query

    def fake_run_query(sql, profile):
        captured["sql"] = sql
        return [{"a": "1"}]

    mod.run_query = fake_run_query
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mod.write_table_to_csv("cat.schema.t", "p", str(Path(tmp) / "o.csv"), where="x > 1")
        assert captured["sql"] == "SELECT * FROM cat.schema.t WHERE x > 1", captured
    finally:
        mod.run_query = original


def test_table_without_three_parts_is_rejected():
    for bad in ("t", "cat.schema", "cat.schema.tbl.extra", "cat..tbl"):
        try:
            mod.write_table_to_csv(bad, "p", "/dev/null")
        except ValueError as exc:
            assert "invalid --table" in str(exc), (bad, exc)
            continue
        raise AssertionError(f"expected ValueError for --table {bad!r}")


def test_hyphenated_table_parts_are_allowed():
    # Real catalog/schema names can contain hyphens (databricks-core's own
    # convention) -- the shape check must not restrict to \w+.
    original = mod.run_query
    mod.run_query = lambda sql, profile: [{"a": "1"}]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            n = mod.write_table_to_csv("acme-prod-sc.data-store.t-outbound", "p", str(Path(tmp) / "o.csv"))
        assert n == 1
    finally:
        mod.run_query = original


def test_empty_result_raises():
    original = mod.run_query
    mod.run_query = lambda sql, profile: []
    try:
        mod.write_table_to_csv("cat.schema.tbl", "p", "/dev/null")
    except ValueError as exc:
        assert "no rows" in str(exc)
        return
    finally:
        mod.run_query = original
    raise AssertionError("expected ValueError for an empty query result")


if __name__ == "__main__":
    test_run_query_normalizes_empty_string_to_none()
    test_run_query_handles_non_json_ddl_success_message()
    test_writes_csv_from_query_rows()
    test_where_clause_is_appended_to_sql()
    test_table_without_three_parts_is_rejected()
    test_hyphenated_table_parts_are_allowed()
    test_empty_result_raises()
    print("ok")
