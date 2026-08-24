"""Confirms formulas that didn't change produce the same numbers as
janrth's originals, and explicitly documents where behavior legitimately
differs (the 2 fixes)."""
import math

import pytest

from replenishment.math_ import normal_quantile, rmse, mae
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import SqrtHorizonSafetyStock, KRmseSafetyStock, KMaeSafetyStock


def test_sqrt_horizon_matches_janrth_formula_shape():
    # janrth: std_dev = rmse * sqrt(horizon); ss = z * std_dev
    forecast = TimeSeries.from_values([10] * 10)
    actuals = TimeSeries.from_values([10, 12, 8, 11, 9, 10, 13, 7, 10, 10])
    factor = 1.65
    strategy = SqrtHorizonSafetyStock(factor=factor)
    ours = strategy.compute(forecast=forecast, actuals=actuals, period=8, lead_time=1, horizon=1, service_level_factor=factor)

    error = rmse([10, 12, 8, 11, 9, 10, 13, 7], [10] * 8)
    janrth_equivalent = factor * error * math.sqrt(2)  # lead_time(1) + horizon(1)
    assert abs(ours - janrth_equivalent) < 1e-9


def test_k_rmse_matches_janrth_formula_shape():
    forecast = TimeSeries.from_values([10] * 10)
    actuals = TimeSeries.from_values([10, 12, 8, 11, 9, 10, 13, 7, 10, 10])
    factor = 1.65
    strategy = KRmseSafetyStock(factor=factor)
    ours = strategy.compute(forecast=forecast, actuals=actuals, period=8, lead_time=1, horizon=1, service_level_factor=factor)
    error = rmse([10, 12, 8, 11, 9, 10, 13, 7], [10] * 8)
    assert abs(ours - factor * error) < 1e-9


def test_k_mae_matches_janrth_formula_shape():
    # janrth: ss = factor * MAE, flat, no horizon scaling -- same shape as k_rmse
    # but using mean absolute error instead of root-mean-square error.
    forecast = TimeSeries.from_values([10] * 10)
    actuals = TimeSeries.from_values([10, 12, 8, 11, 9, 10, 13, 7, 10, 10])
    factor = 1.65
    strategy = KMaeSafetyStock(factor=factor)
    ours = strategy.compute(forecast=forecast, actuals=actuals, period=8, lead_time=1, horizon=1, service_level_factor=factor)
    error = mae([10, 12, 8, 11, 9, 10, 13, 7], [10] * 8)
    assert abs(ours - factor * error) < 1e-9


def test_normal_quantile_matches_janrth_service_levels_normal_quantile():
    # Both implementations use the same Acklam coefficients -- confirms
    # the port didn't introduce a transcription error. Checked against the
    # textbook standard-normal quantiles (not just sign) for real parity
    # coverage -- a "> 0" check alone would pass for almost any positive
    # function and wouldn't catch a transcription bug in the coefficients.
    expected = {
        0.85: 1.0364,
        0.90: 1.2816,
        0.95: 1.6449,
        0.975: 1.9600,
        0.99: 2.3263,
    }
    for p, expected_value in expected.items():
        assert abs(normal_quantile(p) - expected_value) < 1e-4


def test_fill_rate_behavior_intentionally_differs_on_out_of_range_input():
    """Documents fix #1: janrth silently clips here; we raise by default."""
    from replenishment.strategies.safety_stock import FillRateSafetyStock, SafetyStockRangeError
    extreme_actuals = TimeSeries.from_values([10] * 50 + [10, 500])
    forecast = TimeSeries.from_values([10] * 52)
    strategy = FillRateSafetyStock(target_fill_rate=0.999999999)
    with pytest.raises(SafetyStockRangeError):
        strategy.compute(forecast=forecast, actuals=extreme_actuals, period=52, lead_time=0, horizon=1, service_level_factor=0.999999999)
