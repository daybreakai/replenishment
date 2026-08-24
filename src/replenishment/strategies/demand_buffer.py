"""Wraps any SafetyStockStrategy with a trend-chasing uplift: inflate the
buffer when the current forecast is running above a reference baseline.
Ported from janrth's demand_buffer_strength/_reference/_max_multiplier
parameters, which every one of the 7 old policy classes carried and
applied identically -- here it's a decorator instead of a duplicated
parameter set."""
from __future__ import annotations

from dataclasses import dataclass

from replenishment.timeseries import TimeSeries


@dataclass(frozen=True)
class DemandBufferDecorator:
    wrapped: object  # SafetyStockStrategy
    strength: float
    reference: float | None = None
    max_multiplier: float | None = None

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("strength must be non-negative.")
        if self.reference is not None and self.reference <= 0:
            raise ValueError("reference must be positive when provided.")
        if self.max_multiplier is not None and self.max_multiplier < 1:
            raise ValueError("max_multiplier must be at least 1.")

    def compute(self, *, forecast: TimeSeries, actuals, period: int, lead_time: int, horizon: int, service_level_factor: float) -> float:
        base = self.wrapped.compute(
            forecast=forecast, actuals=actuals, period=period,
            lead_time=lead_time, horizon=horizon, service_level_factor=service_level_factor,
        )
        forecast_qty = forecast.sum_over(period, max(horizon, 1))
        reference_qty = self.reference
        if self.strength <= 0 or forecast_qty <= 0 or not reference_qty:
            return base
        uplift = max(0.0, (forecast_qty / reference_qty) - 1.0)
        multiplier = 1.0 + (self.strength * uplift)
        if self.max_multiplier is not None:
            multiplier = min(multiplier, self.max_multiplier)
        return base * max(1.0, multiplier)
