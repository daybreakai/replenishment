import pytest
from replenishment.timeseries import TimeSeries
from replenishment.strategies.multiplier import MultiplierSafetyStockStrategy, NullSafetyStockStrategy

FORECAST = TimeSeries.from_values([10, 10, 10, 10])


def test_multiplier_returns_forecast_times_multiplier_minus_one():
    strategy = MultiplierSafetyStockStrategy(multiplier=1.3)
    # forecast_qty over horizon=1 starting at period=2 is forecast.value_at(2) == 10
    ss = strategy.compute(forecast=FORECAST, actuals=None, period=2, lead_time=0, horizon=1, service_level_factor=1.0)
    assert abs(ss - 10 * (1.3 - 1)) < 1e-9


def test_multiplier_below_one_rejected_at_construction():
    with pytest.raises(ValueError):
        MultiplierSafetyStockStrategy(multiplier=0.8)


def test_multiplier_does_not_require_actuals():
    strategy = MultiplierSafetyStockStrategy(multiplier=1.2)
    ss = strategy.compute(forecast=FORECAST, actuals=None, period=0, lead_time=0, horizon=1, service_level_factor=1.0)
    assert ss >= 0


def test_null_strategy_always_returns_zero():
    strategy = NullSafetyStockStrategy()
    assert strategy.compute(forecast=FORECAST, actuals=None, period=5, lead_time=2, horizon=3, service_level_factor=0.95) == 0.0
