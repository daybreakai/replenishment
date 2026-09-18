"""ponytail self-check for build_segmentation_sql.py -- run `python test_build_segmentation_sql.py`."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_segmentation_sql import (  # noqa: E402
    build_pull_statements, build_write_back_sql, format_pull_sql_for_audit, write_audit_file,
)


def test_build_pull_statements_requires_outbound_shipment():
    try:
        build_pull_statements("cat", "sch", ["t_product"])
    except ValueError as exc:
        assert "t_outbound_shipment" in str(exc)
        return
    raise AssertionError("expected ValueError when t_outbound_shipment is missing")


def test_build_pull_statements_rejects_unknown_table():
    try:
        build_pull_statements("cat", "sch", ["t_outbound_shipment", "t_mystery"])
    except ValueError as exc:
        assert "t_mystery" in str(exc)
        return
    raise AssertionError("expected ValueError for an unregistered table")


def test_pull_sql_references_catalog_and_schema():
    statements = build_pull_statements("acme_prod_sc", "data_store", ["t_outbound_shipment"])
    assert "acme_prod_sc.data_store.t_outbound_shipment" in statements["t_outbound_shipment"]
    audit_text = format_pull_sql_for_audit(statements)
    assert "acme_prod_sc.data_store.t_outbound_shipment" in audit_text


def test_write_back_sql_is_create_and_insert_only():
    records = [{
        "run_timestamp": "2026-01-01T00:00:00Z", "product_id": "A", "strategies_run": "adi_cv2",
        "adi_cv2_class": "smooth", "revenue": 100.0, "abc_class": "A", "xyz_class": "X", "abc_xyz_matrix": "AX",
    }]
    create_sql, insert_sql = build_write_back_sql("acme_prod_sc", "data_store", records)
    assert create_sql.startswith("CREATE TABLE IF NOT EXISTS acme_prod_sc.data_store.t_segmentation")
    assert insert_sql.startswith("INSERT INTO acme_prod_sc.data_store.t_segmentation")
    for forbidden in ("DROP", "MERGE", "DELETE", "OVERWRITE"):
        assert forbidden not in create_sql and forbidden not in insert_sql


def test_write_back_sql_escapes_single_quotes():
    records = [{
        "run_timestamp": "2026-01-01T00:00:00Z", "product_id": "O'Brien", "strategies_run": "adi_cv2",
        "adi_cv2_class": "smooth", "revenue": 1.0, "abc_class": None, "xyz_class": None, "abc_xyz_matrix": None,
    }]
    _, insert_sql = build_write_back_sql("acme_prod_sc", "data_store", records)
    assert "O''Brien" in insert_sql, insert_sql


def test_write_audit_file_rejects_path_traversal_customer():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            write_audit_file("SELECT 1;", "../escape", Path(tmp))
        except ValueError as exc:
            assert "invalid --customer" in str(exc)
            return
    raise AssertionError("expected ValueError for a path-traversal customer name")


def test_write_audit_file_writes_under_customer_dir():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_audit_file("SELECT 1;", "acme", Path(tmp))
        assert path.parent.name == "acme"
        assert path.read_text() == "SELECT 1;"


if __name__ == "__main__":
    test_build_pull_statements_requires_outbound_shipment()
    test_build_pull_statements_rejects_unknown_table()
    test_pull_sql_references_catalog_and_schema()
    test_write_back_sql_is_create_and_insert_only()
    test_write_back_sql_escapes_single_quotes()
    test_write_audit_file_rejects_path_traversal_customer()
    test_write_audit_file_writes_under_customer_dir()
    print("ok")
