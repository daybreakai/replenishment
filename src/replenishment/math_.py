"""Pure math: normal distribution helpers and forecast-error statistics."""
from __future__ import annotations

import math
import statistics


def normal_quantile(p: float) -> float:
    """Approximate the standard normal quantile (inverse CDF)."""
    if not 0.0 < p < 1.0:
        raise ValueError("normal_quantile requires 0 < p < 1.")

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
    )


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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
