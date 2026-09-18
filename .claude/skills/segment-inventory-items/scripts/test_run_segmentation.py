"""ponytail self-check for run_segmentation.py -- run `python test_run_segmentation.py`.
Never calls the real Databricks CLI: fetch_rows/run_query are bypassed by
feeding compute_segments/run() synthetic rows directly, or by monkeypatching
run_query for the dry-run integration test.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_segmentation  # noqa: E402
from run_segmentation import compute_segments  # noqa: E402

_SHIPMENTS = [
    {"product_id": "A", "actual_ship_date": "2026-01-01", "shipped_qty": 10.0},
    {"product_id": "A", "actual_ship_date": "2026-02-01", "shipped_qty": 12.0},
    {"product_id": "B", "actual_ship_date": "2026-01-01", "shipped_qty": 1.0},
    {"product_id": "B", "actual_ship_date": "2026-03-01", "shipped_qty": 500.0},
]
_PRODUCTS = [
    {"product_id": "A", "unit_cost": 5.0, "product_group_id": "g1"},
    {"product_id": "B", "unit_cost": 2.0, "product_group_id": "g1"},
]


def test_compute_segments_adi_cv2_only_needs_no_price():
    records = compute_segments({"t_outbound_shipment": _SHIPMENTS}, ["adi_cv2"])
    by_id = {r["product_id"]: r for r in records}
    assert by_id["A"]["adi_cv2_class"] is not None
    assert by_id["A"]["revenue"] is None  # no t_product supplied, no revenue computed


def test_compute_segments_handles_string_typed_shipped_qty():
    # Discovered against a real Databricks table: the CLI's JSON output
    # stringifies numeric columns (shipped_qty comes back as "211.9", not
    # 211.9) -- run_query() normalizes NULLs but not numeric typing, so
    # compute_segments itself must not assume shipped_qty is already a float.
    rows = {
        "t_outbound_shipment": [
            {"product_id": "A", "actual_ship_date": "2026-01-01", "shipped_qty": "10.0"},
            {"product_id": "A", "actual_ship_date": "2026-02-01", "shipped_qty": "12.0"},
        ],
        "t_product": [{"product_id": "A", "unit_cost": 5.0}],
    }
    records = compute_segments(rows, ["abc_revenue"])
    assert records[0]["revenue"] == 110.0, records


def test_compute_segments_abc_xyz_matrix():
    rows = {"t_outbound_shipment": _SHIPMENTS, "t_product": _PRODUCTS}
    records = compute_segments(rows, ["abc_revenue", "xyz_variability", "abc_xyz_matrix"])
    by_id = {r["product_id"]: r for r in records}
    assert by_id["A"]["abc_class"] in {"A", "B", "C"}
    assert by_id["A"]["abc_xyz_matrix"] == by_id["A"]["abc_class"] + by_id["A"]["xyz_class"]


def test_compute_segments_rejects_null_unit_cost_for_revenue_strategy():
    rows = {"t_outbound_shipment": _SHIPMENTS, "t_product": [{"product_id": "A", "unit_cost": None}]}
    try:
        compute_segments(rows, ["abc_revenue"])
    except ValueError as exc:
        assert "unit_cost" in str(exc)
        return
    raise AssertionError("expected ValueError for a null unit_cost feeding a revenue strategy")


def test_compute_segments_rejects_product_missing_from_t_product_entirely():
    # The gap a missing_cost_ids-only check used to miss: B is shipped but
    # has no t_product row at all (not even a null-cost one) -- must still
    # be caught, not silently dropped from the output.
    rows = {"t_outbound_shipment": _SHIPMENTS, "t_product": [{"product_id": "A", "unit_cost": 5.0}]}
    try:
        compute_segments(rows, ["abc_revenue"])
    except ValueError as exc:
        assert "B" in str(exc) and "unit_cost" in str(exc)
        return
    raise AssertionError("expected ValueError for a product absent from t_product entirely")


_HIERARCHY_PRODUCTS = [
    {"product_id": "A", "unit_cost": 4.0, "product_group_id": "g1"},
    {"product_id": "B", "unit_cost": 4.0, "product_group_id": "g1"},
    {"product_id": "C", "unit_cost": 2.0, "product_group_id": "g2"},
]
_HIERARCHY_SHIPMENTS = [
    {"product_id": "A", "actual_ship_date": "2026-01-01", "shipped_qty": 100.0},
    {"product_id": "B", "actual_ship_date": "2026-01-01", "shipped_qty": 100.0},
    {"product_id": "C", "actual_ship_date": "2026-01-01", "shipped_qty": 100.0},
]


def test_compute_segments_abc_revenue_by_hierarchy_rolls_up_by_group():
    rows = {"t_outbound_shipment": _HIERARCHY_SHIPMENTS, "t_product": _HIERARCHY_PRODUCTS}
    records = compute_segments(rows, ["abc_revenue_by_hierarchy"])
    by_id = {r["product_id"]: r for r in records}
    # A and B share group g1 -> same hierarchy class, computed from their
    # COMBINED revenue (800), not their individual revenue (400 each) --
    # that's the actual bug this fixture targets: before the fix, this
    # strategy never grouped at all and left every field None.
    assert by_id["A"]["abc_hierarchy_class"] == by_id["B"]["abc_hierarchy_class"], by_id
    assert by_id["A"]["abc_hierarchy_class"] is not None, by_id
    # g1 (800) and g2 (200) are different groups -> different buckets.
    assert by_id["A"]["abc_hierarchy_class"] != by_id["C"]["abc_hierarchy_class"], by_id


def test_compute_segments_abc_revenue_by_hierarchy_uses_parent_group_when_present():
    rows = {
        "t_outbound_shipment": _HIERARCHY_SHIPMENTS,
        "t_product": _HIERARCHY_PRODUCTS,
        "t_product_hierarchy": [
            {"product_group_id": "g1", "parent_product_group_id": "p1"},
            {"product_group_id": "g2", "parent_product_group_id": "p1"},
        ],
    }
    records = compute_segments(rows, ["abc_revenue_by_hierarchy"])
    by_id = {r["product_id"]: r for r in records}
    # g1 and g2 now roll up to the same parent p1 -> everyone shares one bucket
    # (whichever letter that merged group computes to; the point being tested
    # is that the parent mapping actually merges them, not the letter itself).
    classes = {by_id["A"]["abc_hierarchy_class"], by_id["B"]["abc_hierarchy_class"], by_id["C"]["abc_hierarchy_class"]}
    assert len(classes) == 1, by_id


def test_run_dry_run_skips_databricks_write_and_writes_audit_file(monkeypatch):
    def fake_run_query(sql, profile):
        if "t_outbound_shipment" in sql:
            return _SHIPMENTS
        if "t_product" in sql:
            return _PRODUCTS
        raise AssertionError(f"unexpected query in dry-run test: {sql}")
    monkeypatch.setattr(run_segmentation, "run_query", fake_run_query)

    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            catalog="acme_prod_sc", schema="data_store", profile="acme-profile", customer="acme",
            strategies=["abc_revenue"], tables=["t_outbound_shipment", "t_product"],
            dry_run=True, out_dir=tmp,
        )
        records, audit_path = run_segmentation.run(args)
        assert len(records) == 2
        assert audit_path.exists()
        assert "CREATE TABLE IF NOT EXISTS acme_prod_sc.data_store.t_segmentation" in audit_path.read_text()


def test_run_rejects_bad_customer_before_calling_databricks(monkeypatch):
    def fail_run_query(sql, profile):
        raise AssertionError("run_query must not be called when --customer fails validation")
    monkeypatch.setattr(run_segmentation, "run_query", fail_run_query)

    args = argparse.Namespace(
        catalog="acme_prod_sc", schema="data_store", profile="acme-profile", customer="../escape",
        strategies=["adi_cv2"], tables=["t_outbound_shipment", "t_product"],
        dry_run=True, out_dir=None,
    )
    try:
        run_segmentation.run(args)
    except ValueError as exc:
        assert "invalid --customer" in str(exc)
        return
    raise AssertionError("expected ValueError for a path-traversal customer name")


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_compute_segments_adi_cv2_only_needs_no_price()
    test_compute_segments_handles_string_typed_shipped_qty()
    test_compute_segments_abc_xyz_matrix()
    test_compute_segments_rejects_null_unit_cost_for_revenue_strategy()
    test_compute_segments_rejects_product_missing_from_t_product_entirely()
    test_compute_segments_abc_revenue_by_hierarchy_rolls_up_by_group()
    test_compute_segments_abc_revenue_by_hierarchy_uses_parent_group_when_present()
    test_run_dry_run_skips_databricks_write_and_writes_audit_file(_FakeMonkeypatch())
    test_run_rejects_bad_customer_before_calling_databricks(_FakeMonkeypatch())
    print("ok")
