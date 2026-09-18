"""ponytail self-check for classify_items.py -- run `python test_classify_items.py`."""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify_items import classify, load_rows  # noqa: E402

FIELDNAMES = [
    "unique_id", "ds", "demand", "forecast", "actuals", "holding_cost_per_unit",
    "stockout_cost_per_unit", "order_cost_per_order", "lead_time", "initial_on_hand",
    "current_stock", "is_forecast",
]


def _write_csv(path: Path, series_by_id: dict[str, list[int]]) -> None:
    rows = []
    for uid, series in series_by_id.items():
        for period, demand in enumerate(series):
            rows.append({
                "unique_id": uid, "ds": f"2024-01-{period + 1:02d}", "demand": demand, "forecast": 10,
                "actuals": demand, "holding_cost_per_unit": 1.0, "stockout_cost_per_unit": 5.0,
                "order_cost_per_order": 0.0, "lead_time": 2, "initial_on_hand": 20,
                "current_stock": 20, "is_forecast": False,
            })
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_smooth_series_classified_smooth():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.csv"
        _write_csv(path, {"A": [10, 11, 9, 10, 10, 11, 9, 10]})
        result = classify(load_rows(str(path)))
        assert result["A"]["demand_class"] == "smooth", result


def test_intermittent_series_classified_intermittent():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.csv"
        _write_csv(path, {"B": [0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0]})
        result = classify(load_rows(str(path)))
        assert result["B"]["demand_class"] == "intermittent", result
        assert result["B"]["zero_share"] > 0.5, result


def test_multiple_unique_ids_are_classified_independently():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.csv"
        _write_csv(path, {
            "smooth-1": [10, 11, 9, 10, 10, 11, 9, 10],
            "sparse-1": [0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
        })
        result = classify(load_rows(str(path)))
        assert set(result) == {"smooth-1", "sparse-1"}, result
        assert result["smooth-1"]["demand_class"] != result["sparse-1"]["demand_class"]


def test_unsupported_extension_is_rejected():
    try:
        load_rows("nope.txt")
    except ValueError as exc:
        assert "nope.txt" in str(exc)
        return
    raise AssertionError("expected ValueError for an unsupported --data extension")


def test_actuals_field_override_matches_backtest_behavior():
    # Same drift concern grid_search.py/backtest.py already guard against:
    # a renamed actuals column must resolve via load_standard_simulation_rows
    # here too, not just for backtest.py's own CLI.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.csv"
        fieldnames = [f if f != "actuals" else "actual_units" for f in FIELDNAMES]
        rows = [{
            "unique_id": "A", "ds": "2024-01-01", "demand": 7, "forecast": 10,
            "actual_units": 7, "holding_cost_per_unit": 1.0, "stockout_cost_per_unit": 5.0,
            "order_cost_per_order": 0.0, "lead_time": 2, "initial_on_hand": 20,
            "current_stock": 20, "is_forecast": False,
        }]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        result = classify(load_rows(str(path), actuals_field="actual_units"))
        assert result["A"]["has_actuals"] is True, result


if __name__ == "__main__":
    test_smooth_series_classified_smooth()
    test_intermittent_series_classified_intermittent()
    test_multiple_unique_ids_are_classified_independently()
    test_unsupported_extension_is_rejected()
    test_actuals_field_override_matches_backtest_behavior()
    print("ok")
