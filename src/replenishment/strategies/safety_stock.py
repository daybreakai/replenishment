"""Safety-stock strategies: how much buffer to add on top of the forecast.

Ported from janrth's policies.py SAFETY_STOCK_METHOD_* branches, split into
one strategy class per method so each is independently testable and the
policy layer never branches on a method-name string.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from replenishment.math_ import rmse as _rmse, mae as _mae, normal_loss, inverse_normal_loss
from replenishment.timeseries import TimeSeries


def _resolve_factor(cls_name: str, factor, k) -> float:
    """factor=/k= alias resolution shared by the flat-multiplier strategies,
    with an error message pointing at the right name instead of a bare
    'unexpected keyword argument'."""
    if factor is not None and k is not None:
        raise TypeError(f"{cls_name} accepts either factor= or k=, not both.")
    resolved = factor if factor is not None else k
    if resolved is None:
        raise TypeError(f"{cls_name} requires factor= (k= also accepted).")
    return resolved


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

    def __init__(self, factor: float | None = None, *, k: float | None = None) -> None:
        object.__setattr__(self, "factor", _resolve_factor("SqrtHorizonSafetyStock", factor, k))

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

    def __init__(self, factor: float | None = None, *, k: float | None = None) -> None:
        object.__setattr__(self, "factor", _resolve_factor("KRmseSafetyStock", factor, k))

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        actual_values, forecast_values = _error_series(forecast, actuals, period)
        return self.factor * _rmse(actual_values, forecast_values)


@dataclass(frozen=True)
class KMaeSafetyStock:
    """SS = factor * MAE, flat, no horizon scaling."""

    factor: float

    def __init__(self, factor: float | None = None, *, k: float | None = None) -> None:
        object.__setattr__(self, "factor", _resolve_factor("KMaeSafetyStock", factor, k))

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        actual_values, forecast_values = _error_series(forecast, actuals, period)
        return self.factor * _mae(actual_values, forecast_values)


class SafetyStockRangeError(ValueError):
    """Raised when a fill-rate target and error distribution combination
    falls outside the range inverse_normal_loss can solve exactly, and
    the strategy was not explicitly told to clip."""

    def __init__(self, *, target_fill_rate: float, mean_demand: float, std_dev: float, bound_hit: float):
        self.target_fill_rate = target_fill_rate
        self.mean_demand = mean_demand
        self.std_dev = std_dev
        self.bound_hit = bound_hit
        super().__init__(
            f"Fill rate {target_fill_rate:.6f} with mean_demand={mean_demand:.4f}, "
            f"std_dev={std_dev:.4f} requires a z outside the solvable range "
            f"(hit bound {bound_hit}). Pass on_out_of_range='clip' to accept "
            f"the boundary value instead of raising."
        )


@dataclass(frozen=True)
class FillRateSafetyStock:
    """SS = z_fill_rate(target) * std_dev, where z_fill_rate is solved by
    inverting the standard normal loss function. Distinguishes fill rate
    from cycle service level, unlike the other strategies here.

    Fix #1 over the original library: rather than silently clipping to
    +/-6 sigma when the (target_fill_rate, mean_demand, std_dev) combination
    has no exact solution in that range, this raises SafetyStockRangeError
    by default so the caller finds out the input was unsolvable instead of
    silently getting a clamped number. Pass on_out_of_range="clip" to opt
    into the old silent-clamp behavior.
    """

    target_fill_rate: float
    on_out_of_range: Literal["raise", "clip"] = "raise"
    lower_bound: float = -6.0
    upper_bound: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_fill_rate < 1.0:
            raise ValueError("target_fill_rate must be in (0, 1).")
        if self.on_out_of_range not in ("raise", "clip"):
            raise ValueError("on_out_of_range must be 'raise' or 'clip'.")

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        actual_values, forecast_values = _error_series(forecast, actuals, period)
        mean_demand = forecast.sum_over(period, max(horizon, 1)) / max(horizon, 1) if period > 0 else 0.0
        std_dev = _rmse(actual_values, forecast_values)
        if std_dev <= 0 or mean_demand <= 0:
            return 0.0

        loss_target = (1.0 - self.target_fill_rate) * mean_demand / std_dev
        # The solvable range for exact inversion is [L(upper_bound), L(lower_bound)]
        # (normal_loss is strictly decreasing), NOT [0, L(lower_bound)] -- loss_target
        # is always > 0 given the guards above, so a "<= 0" check would never fire.
        min_loss = normal_loss(self.upper_bound)
        max_loss = normal_loss(self.lower_bound)
        would_clip_low = loss_target <= min_loss
        would_clip_high = loss_target >= max_loss

        if (would_clip_low or would_clip_high) and self.on_out_of_range == "raise":
            bound = self.upper_bound if would_clip_low else self.lower_bound
            raise SafetyStockRangeError(
                target_fill_rate=self.target_fill_rate,
                mean_demand=mean_demand,
                std_dev=std_dev,
                bound_hit=bound,
            )

        z = inverse_normal_loss(loss_target, lower=self.lower_bound, upper=self.upper_bound)
        return z * std_dev
