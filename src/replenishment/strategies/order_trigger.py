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
class FlatForecastOrderUpToTrigger:
    """Order-up-to with a FLAT forecast target: order_up_to =
    forecast[period + 1] * forecast_horizon + safety_stock.

    Exists because OrderUpToTrigger sums forecast[period+1 .. period+horizon],
    which is only leak-free when the forecast series is a true forward
    forecast fixed at one origin. In backtest calibration the series is
    usually rolling one-step-ahead cross-validation values, where entry
    period+k (k > 1) was produced at origin period+k-1 -- AFTER the order
    decision at `period`. Summing them leaks future demand into the target,
    and the leak grows with model reactivity (a naive forecast becomes a
    near-oracle). This trigger uses only forecast[period + 1] -- the freshest
    value available at decision time -- times the horizon: the classic
    order-up-to arithmetic (forecast_t * cover)."""

    def order_quantity(self, *, inventory_position, period, review_period, forecast, safety_stock, lead_time, forecast_horizon) -> int:
        if review_period > 1 and period % review_period != 0:
            return 0
        target = forecast.value_at(period + 1) * forecast_horizon + safety_stock
        return max(0, math.ceil(target - inventory_position))


@dataclass(frozen=True)
class ReorderPointTrigger:
    def status(self, *, period, forecast, safety_stock, lead_time, forecast_horizon) -> tuple[float, float]:
        """(reorder_point, order_up_to_target) at `period`, independent of
        whether inventory_position has actually crossed reorder_point yet --
        callers that need to know how CLOSE an item is to triggering (e.g.
        pooled/joint replenishment ranking not-yet-triggered items) use
        this; order_quantity uses it too, for one formula in one place."""
        lead_horizon = max(0, lead_time)
        lead_demand = forecast.sum_over(period + 1, lead_horizon) if lead_horizon > 0 else 0
        cycle_stock = forecast.sum_over(period + 1 + lead_horizon, forecast_horizon)
        reorder_point = lead_demand + safety_stock
        return reorder_point, reorder_point + cycle_stock

    def order_quantity(self, *, inventory_position, period, review_period, forecast, safety_stock, lead_time, forecast_horizon) -> int:
        if review_period > 1 and period % review_period != 0:
            return 0
        reorder_point, order_up_to = self.status(
            period=period, forecast=forecast, safety_stock=safety_stock,
            lead_time=lead_time, forecast_horizon=forecast_horizon)
        if inventory_position <= reorder_point:
            return max(0, math.ceil(order_up_to - inventory_position))
        return 0


@dataclass(frozen=True)
class FlatReorderPointTrigger:
    """Reorder-point trigger with a FLAT forecast target -- the same fix
    FlatForecastOrderUpToTrigger applies to OrderUpToTrigger, applied here
    instead: ReorderPointTrigger sums forecast[period+1 .. period+lead_time]
    for lead_demand and forecast[period+1+lead_time .. +forecast_horizon]
    for cycle_stock, which leaks future demand into both the reorder point
    and the order-up-to target when the forecast series is a rolling
    one-step-ahead CV series (see FlatForecastOrderUpToTrigger's docstring
    for the full rationale). This trigger uses only forecast[period + 1] --
    the freshest value available at decision time -- in place of every
    forecast.sum_over(...) call; the reorder-point decision logic (only
    order when inventory_position <= reorder_point) is unchanged."""

    def status(self, *, period, forecast, safety_stock, lead_time, forecast_horizon) -> tuple[float, float]:
        """(reorder_point, order_up_to_target) -- see ReorderPointTrigger.status."""
        flat_forecast = forecast.value_at(period + 1)
        lead_horizon = max(0, lead_time)
        lead_demand = flat_forecast * lead_horizon
        cycle_stock = flat_forecast * forecast_horizon
        reorder_point = lead_demand + safety_stock
        return reorder_point, reorder_point + cycle_stock

    def order_quantity(self, *, inventory_position, period, review_period, forecast, safety_stock, lead_time, forecast_horizon) -> int:
        if review_period > 1 and period % review_period != 0:
            return 0
        reorder_point, order_up_to = self.status(
            period=period, forecast=forecast, safety_stock=safety_stock,
            lead_time=lead_time, forecast_horizon=forecast_horizon)
        if inventory_position <= reorder_point:
            return max(0, math.ceil(order_up_to - inventory_position))
        return 0
