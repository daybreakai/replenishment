import pytest

from replenishment.classification import classify_abc, classify_demand, classify_xyz


def test_empty_history_is_smooth():
    assert classify_demand([]) == "smooth"


def test_all_zeros_is_intermittent():
    assert classify_demand([0, 0, 0, 0]) == "intermittent"


def test_smooth_low_adi_low_cv2():
    # nonzero every period, low variance
    assert classify_demand([10, 11, 9, 10, 10, 9, 11, 10]) == "smooth"


def test_erratic_low_adi_high_cv2():
    # nonzero every period, high variance
    history = [1, 50, 2, 45, 3, 48, 1, 47]
    assert classify_demand(history) == "erratic"


def test_intermittent_high_adi_low_cv2():
    # sparse but uniform nonzero size
    history = [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]
    assert classify_demand(history) == "intermittent"


def test_lumpy_high_adi_high_cv2():
    history = [0, 0, 0, 5, 0, 0, 0, 80, 0, 0, 0, 3]
    assert classify_demand(history) == "lumpy"


def test_single_nonzero_event_is_very_intermittent():
    history = [0] * 20 + [10]
    assert classify_demand(history) in ("intermittent", "lumpy")


def test_classify_abc_splits_by_cumulative_share():
    # one item is 80% of total value -> A alone; next two share the rest.
    values = {"big": 80.0, "mid": 15.0, "small": 5.0}
    result = classify_abc(values)
    assert result["big"] == "A"
    assert result["mid"] == "B"
    assert result["small"] == "C"


def test_classify_abc_all_zero_is_c():
    values = {"a": 0.0, "b": 0.0}
    assert classify_abc(values) == {"a": "C", "b": "C"}


def test_classify_abc_rejects_bad_cutoffs():
    with pytest.raises(ValueError):
        classify_abc({"a": 1.0}, cutoffs=(0.95, 0.8))
    with pytest.raises(ValueError):
        classify_abc({"a": -1.0})


def test_classify_xyz_buckets():
    assert classify_xyz(0.1) == "X"
    assert classify_xyz(0.7) == "Y"
    assert classify_xyz(1.5) == "Z"


def test_classify_xyz_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        classify_xyz(0.5, thresholds=(1.0, 0.5))
    with pytest.raises(ValueError):
        classify_xyz(-0.1)
