import math
import pytest
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import (
    SqrtHorizonSafetyStock, KRmseSafetyStock, KMaeSafetyStock,
)


FORECAST = TimeSeries.from_values([10, 10, 10, 10, 10, 10])
ACTUALS = TimeSeries.from_values([10, 12, 8, 11, 9, 10])


def test_sqrt_horizon_scales_error_by_sqrt_of_horizon():
    strategy = SqrtHorizonSafetyStock(factor=1.65)
    ss_h1 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=1, service_level_factor=1.65)
    ss_h4 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=4, service_level_factor=1.65)
    assert abs(ss_h4 - ss_h1 * math.sqrt(4)) < 1e-9


def test_k_rmse_does_not_scale_with_horizon():
    strategy = KRmseSafetyStock(factor=1.65)
    ss_h1 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=1, service_level_factor=1.65)
    ss_h4 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=4, service_level_factor=1.65)
    assert abs(ss_h1 - ss_h4) < 1e-9


def test_k_mae_uses_mean_absolute_error_not_rmse():
    from replenishment.math_ import mae
    strategy = KMaeSafetyStock(factor=1.0)
    ss = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=1, service_level_factor=1.0)
    expected_mae = mae([10, 12, 8, 11, 9, 10], [10, 10, 10, 10, 10, 10])
    assert abs(ss - expected_mae) < 1e-9


def test_k_rmse_requires_actuals():
    strategy = KRmseSafetyStock(factor=1.65)
    with pytest.raises(ValueError, match="actuals"):
        strategy.compute(forecast=FORECAST, actuals=None, period=6, lead_time=0, horizon=1, service_level_factor=1.65)


def test_safety_stock_zero_at_period_zero_no_history():
    strategy = SqrtHorizonSafetyStock(factor=1.65)
    ss = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=0, lead_time=0, horizon=1, service_level_factor=1.65)
    assert ss == 0.0
