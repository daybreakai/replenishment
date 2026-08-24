"""Safety-stock strategies: how much buffer to add on top of the forecast.

Ported from janrth's policies.py SAFETY_STOCK_METHOD_* branches, split into
one strategy class per method so each is independently testable and the
policy layer never branches on a method-name string.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from replenishment.math_ import rmse as _rmse, mae as _mae
from replenishment.timeseries import TimeSeries


class SafetyStockStrategy(Protocol):
    def compute(
        self,
        *,
        forecast: TimeSeries,
        actuals: TimeSeries | None,
        period: int,
        lead_time: int,
        horizon: int,
        service_level_factor: float,
    ) -> float: ...


def _error_series(forecast: TimeSeries, actuals: TimeSeries, period: int) -> tuple[list[float], list[float]]:
    """Aligned (actuals, forecast) arrays for periods [0, period)."""
    if period <= 0:
        return [], []
    actual_values = [actuals.value_at(i) for i in range(period)]
    forecast_values = [forecast.value_at(i) for i in range(period)]
    return actual_values, forecast_values


def _require_actuals(actuals: TimeSeries | None) -> TimeSeries:
    if actuals is None:
        raise ValueError("This safety-stock strategy requires actuals to compute forecast error.")
    return actuals


@dataclass(frozen=True)
class SqrtHorizonSafetyStock:
    """SS = factor * error * sqrt(lead_time + horizon). janrth's default method."""

    factor: float

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        actual_values, forecast_values = _error_series(forecast, actuals, period)
        error = _rmse(actual_values, forecast_values)
        protection_horizon = lead_time + horizon
        lead_time_factor = math.sqrt(protection_horizon if protection_horizon > 0 else 1)
        return self.factor * error * lead_time_factor


@dataclass(frozen=True)
class KRmseSafetyStock:
    """SS = factor * RMSE, flat, no horizon scaling."""

    factor: float

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        actual_values, forecast_values = _error_series(forecast, actuals, period)
        return self.factor * _rmse(actual_values, forecast_values)


@dataclass(frozen=True)
class KMaeSafetyStock:
    """SS = factor * MAE, flat, no horizon scaling."""

    factor: float

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        actual_values, forecast_values = _error_series(forecast, actuals, period)
        return self.factor * _mae(actual_values, forecast_values)
