"""Multiplier and null safety-stock strategies.

MultiplierSafetyStockStrategy folds janrth's EmpiricalMultiplierPolicy
(order = forecast * multiplier) into the same additive-buffer interface
every other strategy uses: safety_stock = forecast_qty * (multiplier - 1),
so `forecast_qty + safety_stock == forecast_qty * multiplier`.

NullSafetyStockStrategy is for forecasts that already embed their own
buffer (e.g. a percentile forecast) -- no separate policy class needed
for that case, just compose ReplenishmentPolicy with this strategy.
"""
from __future__ import annotations

from dataclasses import dataclass

from replenishment.timeseries import TimeSeries


@dataclass(frozen=True)
class MultiplierSafetyStockStrategy:
    multiplier: float

    def __post_init__(self) -> None:
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0 (a multiplier below 1 would reduce, not buffer, the order).")

    def compute(self, *, forecast: TimeSeries, actuals, period: int, lead_time: int, horizon: int, service_level_factor: float) -> float:
        forecast_qty = forecast.sum_over(period, max(horizon, 1))
        return forecast_qty * (self.multiplier - 1.0)


@dataclass(frozen=True)
class NullSafetyStockStrategy:
    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        return 0.0
