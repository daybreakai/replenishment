import pytest
from replenishment.timeseries import TimeSeries


def test_from_values_indexes_directly():
    ts = TimeSeries.from_values([10, 12, 8])
    assert ts.value_at(0) == 10
    assert ts.value_at(2) == 8


def test_from_values_extends_last_value_past_end():
    ts = TimeSeries.from_values([10, 12, 8])
    assert ts.value_at(5) == 8


def test_from_values_rejects_negative_period():
    ts = TimeSeries.from_values([10, 12, 8])
    with pytest.raises(IndexError):
        ts.value_at(-1)


def test_from_values_empty_raises_on_any_lookup():
    ts = TimeSeries.from_values([])
    with pytest.raises(IndexError):
        ts.value_at(0)


def test_from_callable_delegates():
    ts = TimeSeries.from_callable(lambda period: period * 2)
    assert ts.value_at(3) == 6


def test_sum_over_horizon():
    ts = TimeSeries.from_values([10, 12, 8, 9])
    assert ts.sum_over(1, 3) == 12 + 8 + 9


def test_sum_over_zero_horizon_is_zero():
    ts = TimeSeries.from_values([10, 12, 8])
    assert ts.sum_over(0, 0) == 0.0
