import pytest

from replenishment.classification import classify_demand


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
