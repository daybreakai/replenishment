"""ponytail self-check for derive_standard_rows.py -- run
`python test_derive_standard_rows.py`. Never calls the real Databricks CLI:
monkeypatches run_query with fixed fixtures matching real Hasbro-shaped
tables (t_outbound_shipment/t_product/t_vendor_lead_time)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import derive_standard_rows as mod  # noqa: E402
from derive_standard_rows import derive_rows  # noqa: E402

_SHIPMENTS = [
    {"product_id": "A", "actual_ship_date": "2026-01-01", "shipped_qty": "100.0"},
    {"product_id": "A", "actual_ship_date": "2026-02-01", "shipped_qty": "110.0"},
    {"product_id": "A", "actual_ship_date": "2026-03-01", "shipped_qty": "90.0"},
    {"product_id": "A", "actual_ship_date": "2026-04-01", "shipped_qty": "105.0"},
]
_PRODUCTS = [{"product_id": "A", "unit_cost": "10.0"}]
_LEAD_TIMES = [{"product_id": "A", "lead_time_days": "30"}]


def _fake_run_query(sql, profile):
    if "t_outbound_shipment" in sql:
        return _SHIPMENTS
    if "t_product" in sql:
        return _PRODUCTS
    if "t_vendor_lead_time" in sql:
        return _LEAD_TIMES
    raise AssertionError(f"unexpected query: {sql}")


def test_derive_rows_computes_cost_from_real_unit_cost(monkeypatch):
    monkeypatch.setattr(mod, "run_query", _fake_run_query)
    rows = derive_rows("cat", "sch", "p", holding_cost_percent=0.15, stockout_cost_multiplier=0.40, order_cost=0.0)
    assert rows[0]["holding_cost_per_unit"] == 1.5, rows[0]
    assert rows[0]["stockout_cost_per_unit"] == 4.0, rows[0]
    assert rows[0]["order_cost_per_order"] == 0.0, rows[0]
    assert rows[0]["lead_time"] == 1, rows[0]  # 30 days -> 1 month, monthly-grain data


def test_derive_rows_forecast_is_trailing_moving_average(monkeypatch):
    monkeypatch.setattr(mod, "run_query", _fake_run_query)
    rows = derive_rows("cat", "sch", "p", forecast_window=3)
    # period 0: no history yet -> forecast falls back to its own demand (100)
    assert rows[0]["forecast"] == 100, rows[0]
    # period 3 (4th row): trailing 3 of [100, 110, 90] -> mean 100
    assert rows[3]["forecast"] == 100, rows[3]


def test_derive_rows_rejects_missing_unit_cost(monkeypatch):
    def fake(sql, profile):
        if "t_product" in sql:
            return [{"product_id": "A", "unit_cost": None}]
        return _fake_run_query(sql, profile)
    monkeypatch.setattr(mod, "run_query", fake)
    try:
        derive_rows("cat", "sch", "p")
    except ValueError as exc:
        assert "unit_cost" in str(exc)
        return
    raise AssertionError("expected ValueError for a missing unit_cost")


def test_derive_rows_rejects_missing_lead_time(monkeypatch):
    def fake(sql, profile):
        if "t_vendor_lead_time" in sql:
            return []
        return _fake_run_query(sql, profile)
    monkeypatch.setattr(mod, "run_query", fake)
    try:
        derive_rows("cat", "sch", "p")
    except ValueError as exc:
        assert "lead_time_days" in str(exc)
        return
    raise AssertionError("expected ValueError for a missing lead_time_days")


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_derive_rows_computes_cost_from_real_unit_cost(_FakeMonkeypatch())
    test_derive_rows_forecast_is_trailing_moving_average(_FakeMonkeypatch())
    test_derive_rows_rejects_missing_unit_cost(_FakeMonkeypatch())
    test_derive_rows_rejects_missing_lead_time(_FakeMonkeypatch())
    print("ok")
