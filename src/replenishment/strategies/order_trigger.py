"""Order triggers: periodic order-up-to vs continuous-review reorder point.
Ported from janrth's order_quantity_for() bodies, which were duplicated
verbatim across 5 (order-up-to) and 3 (ROP) of the old policy classes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from replenishment.timeseries import TimeSeries


class OrderTrigger(Protocol):
    def order_quantity(
        self, *, inventory_position: int, period: int, review_period: int,
        forecast: TimeSeries, safety_stock: float, lead_time: int, forecast_horizon: int,
    ) -> int: ...


@dataclass(frozen=True)
class OrderUpToTrigger:
    def order_quantity(self, *, inventory_position, period, review_period, forecast, safety_stock, lead_time, forecast_horizon) -> int:
        if review_period > 1 and period % review_period != 0:
            return 0
        start_period = period + max(1, lead_time)
        forecast_qty = forecast.sum_over(start_period, forecast_horizon)
        target = forecast_qty + safety_stock
        return max(0, math.ceil(target - inventory_position))


@dataclass(frozen=True)
class ReorderPointTrigger:
    def order_quantity(self, *, inventory_position, period, review_period, forecast, safety_stock, lead_time, forecast_horizon) -> int:
        if review_period > 1 and period % review_period != 0:
            return 0
        lead_horizon = max(0, lead_time)
        lead_demand = forecast.sum_over(period + 1, lead_horizon) if lead_horizon > 0 else 0
        cycle_stock = forecast.sum_over(period + 1 + lead_horizon, forecast_horizon)
        reorder_point = lead_demand + safety_stock
        order_up_to = reorder_point + cycle_stock
        if inventory_position <= reorder_point:
            return max(0, math.ceil(order_up_to - inventory_position))
        return 0
