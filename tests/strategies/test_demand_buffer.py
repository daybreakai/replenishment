import pytest
from replenishment.timeseries import TimeSeries
from replenishment.strategies.multiplier import NullSafetyStockStrategy
from replenishment.strategies.safety_stock import SqrtHorizonSafetyStock
from replenishment.strategies.demand_buffer import DemandBufferDecorator

FORECAST = TimeSeries.from_values([10, 10, 10, 10, 20])  # spike at period 4
ACTUALS = TimeSeries.from_values([10, 10, 10, 10, 20])


def test_no_uplift_when_strength_is_zero():
    base = SqrtHorizonSafetyStock(factor=1.65)
    decorated = DemandBufferDecorator(wrapped=base, strength=0.0, reference=10.0)
    plain = base.compute(forecast=FORECAST, actuals=ACTUALS, period=4, lead_time=0, horizon=1, service_level_factor=1.65)
    buffered = decorated.compute(forecast=FORECAST, actuals=ACTUALS, period=4, lead_time=0, horizon=1, service_level_factor=1.65)
    assert abs(plain - buffered) < 1e-9


def test_uplift_scales_with_forecast_above_reference():
    base = NullSafetyStockStrategy()
    decorated = DemandBufferDecorator(wrapped=base, strength=1.0, reference=10.0)
    # forecast at period 4, horizon 1 is 20 -- double the reference of 10
    # uplift = strength * (20/10 - 1) = 1.0, multiplier = 1 + 1.0 = 2.0
    # base safety stock is 0 (Null), so decorated is also 0 * multiplier = 0
    # -- use a non-null base to observe the multiplier's effect instead:
    weighted = DemandBufferDecorator(wrapped=SqrtHorizonSafetyStock(factor=1.0), strength=1.0, reference=10.0)
    plain = SqrtHorizonSafetyStock(factor=1.0).compute(forecast=FORECAST, actuals=ACTUALS, period=4, lead_time=0, horizon=1, service_level_factor=1.0)
    buffered = weighted.compute(forecast=FORECAST, actuals=ACTUALS, period=4, lead_time=0, horizon=1, service_level_factor=1.0)
    assert abs(buffered - plain * 2.0) < 1e-6


def test_max_multiplier_caps_the_uplift():
    weighted = DemandBufferDecorator(
        wrapped=SqrtHorizonSafetyStock(factor=1.0), strength=10.0, reference=10.0, max_multiplier=1.5,
    )
    plain = SqrtHorizonSafetyStock(factor=1.0).compute(forecast=FORECAST, actuals=ACTUALS, period=4, lead_time=0, horizon=1, service_level_factor=1.0)
    buffered = weighted.compute(forecast=FORECAST, actuals=ACTUALS, period=4, lead_time=0, horizon=1, service_level_factor=1.0)
    assert abs(buffered - plain * 1.5) < 1e-6


def test_negative_strength_rejected():
    with pytest.raises(ValueError):
        DemandBufferDecorator(wrapped=NullSafetyStockStrategy(), strength=-0.1)
