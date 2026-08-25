import pytest
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import FillRateSafetyStock, KRmseSafetyStock, SqrtHorizonSafetyStock
from replenishment.strategies.multiplier import NullSafetyStockStrategy
from replenishment.strategies.demand_buffer import DemandBufferDecorator
from replenishment.strategies.order_trigger import OrderUpToTrigger, ReorderPointTrigger
from replenishment.policy import ReplenishmentPolicy


class FakeState:
    def __init__(self, period, inventory_position):
        self.period = period
        self.inventory_position = inventory_position


FORECAST = TimeSeries.from_values([10] * 20)
ACTUALS = TimeSeries.from_values([10, 12, 8, 11, 9, 10] + [10] * 14)


def test_order_up_to_classmethod_builds_a_working_policy():
    policy = ReplenishmentPolicy.order_up_to(
        forecast=FORECAST, actuals=ACTUALS,
        safety_stock=SqrtHorizonSafetyStock(factor=1.65),
        lead_time=2, forecast_horizon=1,
    )
    qty = policy.order_quantity_for(FakeState(period=6, inventory_position=5))
    assert qty >= 0


def test_reorder_point_classmethod_builds_a_working_policy():
    policy = ReplenishmentPolicy.reorder_point(
        forecast=FORECAST, actuals=ACTUALS,
        safety_stock=SqrtHorizonSafetyStock(factor=1.65),
        lead_time=2, forecast_horizon=3,
    )
    qty = policy.order_quantity_for(FakeState(period=6, inventory_position=0))
    assert qty > 0


def test_null_safety_stock_policy_orders_forecast_only():
    policy = ReplenishmentPolicy.order_up_to(
        forecast=FORECAST, actuals=None,
        safety_stock=NullSafetyStockStrategy(),
        lead_time=0, forecast_horizon=1,
    )
    qty = policy.order_quantity_for(FakeState(period=3, inventory_position=0))
    assert qty == 10  # exactly the forecast, no buffer


def test_moq_floors_a_positive_order_up_to_the_minimum():
    policy = ReplenishmentPolicy.order_up_to(
        forecast=FORECAST, actuals=None,
        safety_stock=NullSafetyStockStrategy(),
        lead_time=0, forecast_horizon=1, moq=25,
    )
    # gap to target is 10 - 5 = 5, but MoQ floors it to 25
    qty = policy.order_quantity_for(FakeState(period=3, inventory_position=5))
    assert qty == 25


def test_moq_rounds_a_positive_order_up_to_the_nearest_multiple():
    policy = ReplenishmentPolicy.order_up_to(
        forecast=FORECAST, actuals=None,
        safety_stock=NullSafetyStockStrategy(),
        lead_time=0, forecast_horizon=1, moq=4,
    )
    # gap to target is 10 - 0 = 10; pack size 4 -> next multiple is 12, not 10
    qty = policy.order_quantity_for(FakeState(period=3, inventory_position=0))
    assert qty == 12


def test_moq_never_turns_a_no_order_decision_into_an_order():
    policy = ReplenishmentPolicy.order_up_to(
        forecast=FORECAST, actuals=None,
        safety_stock=NullSafetyStockStrategy(),
        lead_time=0, forecast_horizon=1, moq=25,
    )
    # inventory already at target -> trigger says 0 -> stays 0 despite MoQ
    qty = policy.order_quantity_for(FakeState(period=3, inventory_position=10))
    assert qty == 0


def test_moq_respects_review_period_gating():
    policy = ReplenishmentPolicy.order_up_to(
        forecast=FORECAST, actuals=None,
        safety_stock=NullSafetyStockStrategy(),
        lead_time=0, forecast_horizon=1, review_period=2, moq=25,
    )
    # off-review period -> trigger returns 0 -> MoQ must not resurrect it
    qty = policy.order_quantity_for(FakeState(period=3, inventory_position=0))
    assert qty == 0


def test_moq_leaves_orders_already_at_a_multiple_untouched():
    policy = ReplenishmentPolicy.order_up_to(
        forecast=FORECAST, actuals=None,
        safety_stock=NullSafetyStockStrategy(),
        lead_time=0, forecast_horizon=1, moq=5,
    )
    # gap to target is 10, already a multiple of 5 -> unchanged
    qty = policy.order_quantity_for(FakeState(period=3, inventory_position=0))
    assert qty == 10


def test_construction_rejects_moq_below_one():
    with pytest.raises(ValueError, match="moq"):
        ReplenishmentPolicy(
            forecast=FORECAST, safety_stock=NullSafetyStockStrategy(),
            trigger=OrderUpToTrigger(), moq=0,
        )


def test_construction_rejects_negative_lead_time():
    with pytest.raises(ValueError):
        ReplenishmentPolicy(
            forecast=FORECAST, safety_stock=NullSafetyStockStrategy(),
            trigger=OrderUpToTrigger(), lead_time=-1,
        )


def test_construction_rejects_rmse_strategy_without_actuals():
    with pytest.raises(ValueError, match="actuals"):
        ReplenishmentPolicy.order_up_to(
            forecast=FORECAST, actuals=None,
            safety_stock=KRmseSafetyStock(factor=1.65),
            lead_time=0, forecast_horizon=1,
        )


def test_construction_rejects_sqrt_horizon_strategy_without_actuals():
    with pytest.raises(ValueError, match="actuals"):
        ReplenishmentPolicy.order_up_to(
            forecast=FORECAST, actuals=None,
            safety_stock=SqrtHorizonSafetyStock(factor=1.65),
            lead_time=0, forecast_horizon=1,
        )


def test_construction_rejects_fill_rate_strategy_without_actuals():
    with pytest.raises(ValueError, match="actuals"):
        ReplenishmentPolicy.order_up_to(
            forecast=FORECAST, actuals=None,
            safety_stock=FillRateSafetyStock(target_fill_rate=0.95),
            lead_time=0, forecast_horizon=1,
        )


def test_construction_rejects_decorated_strategy_without_actuals():
    with pytest.raises(ValueError, match="actuals"):
        ReplenishmentPolicy.order_up_to(
            forecast=FORECAST, actuals=None,
            safety_stock=DemandBufferDecorator(
                wrapped=KRmseSafetyStock(factor=1.65), strength=0.5, reference=10.0,
            ),
            lead_time=0, forecast_horizon=1,
        )
