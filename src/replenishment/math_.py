"""Pure math: normal distribution helpers and forecast-error statistics."""
from __future__ import annotations

import math
import statistics


_STANDARD_NORMAL = statistics.NormalDist()


def normal_quantile(p: float) -> float:
    """Standard normal quantile (inverse CDF)."""
    if not 0.0 < p < 1.0:
        raise ValueError("normal_quantile requires 0 < p < 1.")
    return _STANDARD_NORMAL.inv_cdf(p)


def normal_pdf(z: float) -> float:
    return _STANDARD_NORMAL.pdf(z)


def normal_cdf(z: float) -> float:
    return _STANDARD_NORMAL.cdf(z)


def normal_loss(z: float) -> float:
    """Standard normal unit loss function L(z) = phi(z) - z*(1-Phi(z))."""
    return normal_pdf(z) - z * (1.0 - normal_cdf(z))


def inverse_normal_loss(value: float, *, lower: float = -6.0, upper: float = 6.0) -> float:
    """Invert normal_loss via bisection. Range is caller-controlled — see
    FillRateSafetyStock, which does NOT rely on this function's default
    silent-clamp behavior for its default configuration."""
    if value <= 0:
        return upper
    max_loss = normal_loss(lower)
    if value >= max_loss:
        return lower
    lo, hi = lower, upper
    for _ in range(60):
        mid = (lo + hi) / 2.0
        loss = normal_loss(mid)
        if loss > value:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def rmse(actuals: list[float], forecasts: list[float]) -> float:
    count = min(len(actuals), len(forecasts))
    if count <= 0:
        return 0.0
    errors = [actuals[i] - forecasts[i] for i in range(count)]
    return math.sqrt(statistics.fmean(e ** 2 for e in errors))


def mae(actuals: list[float], forecasts: list[float]) -> float:
    count = min(len(actuals), len(forecasts))
    if count <= 0:
        return 0.0
    errors = [abs(actuals[i] - forecasts[i]) for i in range(count)]
    return statistics.fmean(errors)
