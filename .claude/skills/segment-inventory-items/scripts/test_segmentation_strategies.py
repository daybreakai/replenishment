"""ponytail self-check for segmentation_strategies.py -- run `python test_segmentation_strategies.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from segmentation_strategies import SEGMENTATION_STRATEGIES, abc_bucket, abc_xyz_matrix, adi_cv2_class, xyz_bucket  # noqa: E402


def test_adi_cv2_class_matches_classify_demand_smooth_series():
    assert adi_cv2_class([10, 11, 9, 10, 10, 11, 9, 10]) == "smooth"


def test_abc_bucket_top_item_is_a_bottom_item_is_c():
    revenue = {"big": 800.0, "mid": 150.0, "small": 50.0}
    buckets = abc_bucket(revenue)
    assert buckets["big"] == "A", buckets
    assert buckets["small"] == "C", buckets


def test_abc_bucket_handles_zero_total_revenue():
    buckets = abc_bucket({"a": 0.0, "b": 0.0})
    assert set(buckets.values()) == {"C"}, buckets  # share defaults to 1.0 when total is 0 -> lowest bucket


def test_xyz_bucket_stable_series_is_x_volatile_series_is_z():
    buckets = xyz_bucket({
        "stable": [100, 101, 99, 100, 100],
        "volatile": [10, 200, 5, 300, 1],
    })
    assert buckets["stable"] == "X", buckets
    assert buckets["volatile"] == "Z", buckets


def test_abc_xyz_matrix_concatenates_labels_for_ids_in_both():
    matrix = abc_xyz_matrix({"a": "A", "b": "B"}, {"a": "X", "c": "Z"})
    assert matrix == {"a": "AX"}, matrix


def test_registry_entries_all_require_outbound_shipment():
    for name, spec in SEGMENTATION_STRATEGIES.items():
        assert "t_outbound_shipment" in spec["required_tables"], name


if __name__ == "__main__":
    test_adi_cv2_class_matches_classify_demand_smooth_series()
    test_abc_bucket_top_item_is_a_bottom_item_is_c()
    test_abc_bucket_handles_zero_total_revenue()
    test_xyz_bucket_stable_series_is_x_volatile_series_is_z()
    test_abc_xyz_matrix_concatenates_labels_for_ids_in_both()
    test_registry_entries_all_require_outbound_shipment()
    print("ok")
