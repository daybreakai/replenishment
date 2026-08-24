import pytest
from replenishment.simulation import simulate_replenishment
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import SqrtHorizonSafetyStock
from replenishment.policy import ReplenishmentPolicy


def test_simulate_replenishment_never_goes_negative_on_hand_with_enough_lead_time_buffer():
    forecast = TimeSeries.from_values([10] * 30)
    actuals = TimeSeries.from_values([10] * 30)
    policy = ReplenishmentPolicy.order_up_to(
        forecast=forecast, actuals=actuals,
        safety_stock=SqrtHorizonSafetyStock(factor=1.65),
        lead_time=2, forecast_horizon=1,
    )
    result = simulate_replenishment(
        periods=20, demand=[10] * 20, initial_on_hand=50, lead_time=2, policy=policy,
    )
    assert all(snap.ending_on_hand >= 0 for snap in result.snapshots)


def test_simulate_replenishment_rejects_non_positive_periods():
    forecast = TimeSeries.from_values([10])
    policy = ReplenishmentPolicy.order_up_to(forecast=forecast, safety_stock=_null(), lead_time=0)
    with pytest.raises(ValueError):
        simulate_replenishment(periods=0, demand=[10], initial_on_hand=0, lead_time=0, policy=policy)


def _null():
    from replenishment.strategies.multiplier import NullSafetyStockStrategy
    return NullSafetyStockStrategy()


def test_summary_reports_total_cost_as_sum_of_components():
    forecast = TimeSeries.from_values([10] * 10)
    policy = ReplenishmentPolicy.order_up_to(forecast=forecast, safety_stock=_null(), lead_time=0)
    result = simulate_replenishment(
        periods=10, demand=[10] * 10, initial_on_hand=0, lead_time=0, policy=policy,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0, order_cost_per_order=2.0,
    )
    s = result.summary
    assert abs(s.total_cost - (s.holding_cost + s.stockout_cost + s.ordering_cost)) < 1e-9


def test_snapshot_backorders_reflects_unmet_demand():
    forecast = TimeSeries.from_values([0] * 5)
    policy = ReplenishmentPolicy.order_up_to(forecast=forecast, safety_stock=_null(), lead_time=0)
    result = simulate_replenishment(
        periods=5, demand=[10, 10, 10, 10, 10], initial_on_hand=0, lead_time=0, policy=policy,
    )
    assert all(snap.backorders == 10 for snap in result.snapshots)


def test_simulate_replenishment_rejects_periods_exceeding_demand_length():
    forecast = TimeSeries.from_values([10] * 5)
    policy = ReplenishmentPolicy.order_up_to(forecast=forecast, safety_stock=_null(), lead_time=0)
    with pytest.raises(ValueError):
        simulate_replenishment(periods=10, demand=[10] * 5, initial_on_hand=0, lead_time=0, policy=policy)
