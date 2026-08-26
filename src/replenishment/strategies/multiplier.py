"""Multiplier and null safety-stock strategies.

MultiplierSafetyStockStrategy folds janrth's EmpiricalMultiplierPolicy
(order = forecast * multiplier) into the same additive-buffer interface
every other strategy uses: safety_stock = forecast_qty * (multiplier - 1),
so `forecast_qty + safety_stock == forecast_qty * multiplier`.

NullSafetyStockStrategy is for forecasts that already embed their own
buffer (e.g. a percentile forecast) -- no separate policy class needed
for that case, just compose ReplenishmentPolicy with this strategy.
Its `value` field defaults to 0.0 ("no extra buffer") but also serves as
a general frozen-constant strategy: pass a precomputed value to carry a
correctly-computed safety-stock number from one window's computation into
another window's policy (e.g. backtest -> evaluation), without pairing
that window's forecast against a different window's actuals.
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
    value: float = 0.0

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        return self.value
