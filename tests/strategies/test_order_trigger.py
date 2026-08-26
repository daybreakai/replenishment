from replenishment.timeseries import TimeSeries
from replenishment.strategies.order_trigger import OrderUpToTrigger, ReorderPointTrigger

FORECAST = TimeSeries.from_values([10] * 20)


def test_order_up_to_orders_forecast_plus_safety_stock_minus_position():
    trigger = OrderUpToTrigger()
    qty = trigger.order_quantity(
        inventory_position=15, period=5, review_period=1,
        forecast=FORECAST, safety_stock=5.0, lead_time=2, forecast_horizon=1,
    )
    # start_period = 5 + lead_time(2) = 7; forecast_qty over horizon 1 = 10
    # target = 10 + 5 = 15; order = max(0, 15 - 15) = 0
    assert qty == 0


def test_order_up_to_skips_non_review_periods():
    trigger = OrderUpToTrigger()
    qty = trigger.order_quantity(
        inventory_position=0, period=1, review_period=3,
        forecast=FORECAST, safety_stock=5.0, lead_time=0, forecast_horizon=1,
    )
    assert qty == 0  # period 1 is not a multiple of review_period 3


def test_reorder_point_triggers_only_below_rop():
    trigger = ReorderPointTrigger()
    qty_above_rop = trigger.order_quantity(
        inventory_position=100, period=0, review_period=1,
        forecast=FORECAST, safety_stock=5.0, lead_time=2, forecast_horizon=3,
    )
    assert qty_above_rop == 0

    qty_below_rop = trigger.order_quantity(
        inventory_position=0, period=0, review_period=1,
        forecast=FORECAST, safety_stock=5.0, lead_time=2, forecast_horizon=3,
    )
    # lead_demand (2 periods) = 20; cycle_stock (3 periods after) = 30
    # reorder_point = 20 + 5 = 25; order_up_to = 25 + 30 = 55; order = 55 - 0 = 55
    assert qty_below_rop == 55


class TestFlatForecastOrderUpToTrigger:
    def test_target_is_next_forecast_times_horizon(self):
        from replenishment.strategies.order_trigger import FlatForecastOrderUpToTrigger
        trigger = FlatForecastOrderUpToTrigger()
        forecast = TimeSeries.from_values([0, 10, 999, 999, 999])
        qty = trigger.order_quantity(
            inventory_position=5, period=0, review_period=1,
            forecast=forecast, safety_stock=2.0, lead_time=1, forecast_horizon=3)
        # target = forecast[1] * 3 + 2 = 32; qty = 32 - 5 = 27.
        # The 999s at periods 2..4 (future-origin values under rolling
        # one-step CV) must NOT enter the target — that is the leak this
        # trigger exists to close.
        assert qty == 27

    def test_review_gate_still_applies(self):
        from replenishment.strategies.order_trigger import FlatForecastOrderUpToTrigger
        trigger = FlatForecastOrderUpToTrigger()
        forecast = TimeSeries.from_values([0, 10, 10, 10])
        qty = trigger.order_quantity(
            inventory_position=0, period=1, review_period=2,
            forecast=forecast, safety_stock=0.0, lead_time=1, forecast_horizon=2)
        assert qty == 0
