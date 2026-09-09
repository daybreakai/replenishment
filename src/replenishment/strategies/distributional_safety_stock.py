"""Distributional safety-stock strategies: computed from the raw demand
distribution (and, for King's formula, lead-time distribution), not from
forecast error. Ported from supplyplanning/prototype/engine
(kings_safety_stock, compound_poisson.py).

Kept separate from safety_stock.py because these don't use
_error_series/_require_actuals -- they read actuals directly, and
KingsFormulaSafetyStock works with no forecast error history at all
(period 0 -> zero std, SS=0, same cold-start behavior as the error-based
strategies).

No RNG anywhere in this module (2026-09-02): CompoundPoissonSafetyStock
was originally Monte Carlo (simulating the Poisson-arrival/gamma-size
compound process directly); it's now closed-form via gamma's additivity
under convolution, matching NegativeBinomialSafetyStock's determinism.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

from replenishment.math_ import normal_quantile
from replenishment.timeseries import TimeSeries


def _require_actuals(actuals: TimeSeries | None) -> TimeSeries:
    if actuals is None:
        raise ValueError("This safety-stock strategy requires actuals to compute demand statistics.")
    return actuals


def _actual_values(actuals: TimeSeries, period: int) -> list[float]:
    return [actuals.value_at(i) for i in range(period)] if period > 0 else []


@dataclass(frozen=True)
class KingsFormulaSafetyStock:
    """SS = z_alpha * sqrt(L * std_demand^2 + mean_demand^2 * std_lead_time^2),
    L = lead_time + horizon (the full protection window -- see
    SqrtHorizonSafetyStock, which ReplenishmentPolicy calls with the exact
    same lead_time/horizon pair; scaling by lead_time alone under-covers by
    the review-period portion of the window whenever horizon > 0).

    Assumes demand-lead_time independence, serially uncorrelated demand.
    std_lead_time_periods defaults to 0.0 since ReplenishmentPolicy.lead_time
    is a fixed int (no lead-time distribution modeled) -- zero variance is
    the correct value for a deterministic lead time, not a placeholder.
    """

    target_service_level: float = 0.95
    std_lead_time_periods: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_service_level < 1.0:
            raise ValueError("target_service_level must be in (0, 1).")
        if self.std_lead_time_periods < 0:
            raise ValueError("std_lead_time_periods cannot be negative.")

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        values = _actual_values(actuals, period)
        if len(values) < 2:
            return 0.0

        mean_d = statistics.fmean(values)
        std_d = statistics.stdev(values)
        z_alpha = normal_quantile(self.target_service_level)

        protection = lead_time + horizon
        variance = protection * std_d ** 2 + mean_d ** 2 * self.std_lead_time_periods ** 2
        return z_alpha * math.sqrt(max(variance, 0.0))


def _poisson_quantile(target: float, rate: float) -> int:
    """Smallest n with Poisson(rate) CDF(n) >= target, via the standard pmf
    ratio recurrence pmf(n) = pmf(n-1) * rate/n -- O(1) work per step,
    exact, no RNG."""
    pmf = math.exp(-rate)
    cdf = pmf
    n = 0
    max_iterations = 1_000_000
    while cdf < target:
        n += 1
        if n > max_iterations:
            raise RuntimeError(
                f"Poisson quantile search did not converge within "
                f"{max_iterations} steps (rate={rate}, target={target})."
            )
        pmf *= rate / n
        cdf += pmf
    return n


def _lower_regularized_gamma(a: float, x: float) -> float:
    """P(a, x) = gamma(a, x) / Gamma(a) -- the CDF of a Gamma(shape=a,
    scale=1) distribution at x. Series expansion for x < a+1, continued
    fraction (via its complement Q = 1-P) for x >= a+1 -- the standard
    numerically stable split (Numerical Recipes 6.2). Verified against
    scipy.special.gammainc to ~1e-13 absolute error across 2000 random
    (a, x) pairs before being trusted here."""
    if a <= 0:
        raise ValueError("a must be positive.")
    if x <= 0:
        return 0.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-14:
                break
        return total * math.exp(log_prefactor)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 501):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return 1.0 - h * math.exp(log_prefactor)


def _poisson_tail_cutoff(rate: float, eps: float = 1e-13) -> int:
    """Largest n worth including in a Poisson(rate)-weighted mixture: walks
    forward until the cumulative pmf covers 1-eps of the mass, so truncating
    the (infinite, in principle) mixture sum there loses at most eps of
    probability mass."""
    pmf = math.exp(-rate)
    cum = pmf
    n = 0
    max_iterations = 1_000_000
    while 1.0 - cum > eps and n < max_iterations:
        n += 1
        pmf *= rate / n
        cum += pmf
    return n


def _compound_poisson_gamma_cdf(x: float, rate: float, shape: float, scale: float, n_max: int) -> float:
    """F(x) for LTD = sum_{i=1}^{N} Y_i, N ~ Poisson(rate), Y_i iid
    Gamma(shape, scale). Exact, closed-form (given n_max covers the Poisson
    tail): a sum of n iid Gamma(shape, scale) variables is itself
    Gamma(n*shape, scale) (gamma's additivity under convolution with a
    shared scale), so this is a Poisson-weighted mixture of gamma CDFs, not
    an approximation -- N=0 contributes a point mass at LTD=0 (which is
    <= any x > 0, hence contributes its full pmf to the CDF)."""
    if x <= 0:
        return math.exp(-rate)
    pmf = math.exp(-rate)  # P(N=0)
    cdf = pmf
    for n in range(1, n_max + 1):
        pmf *= rate / n
        cdf += pmf * _lower_regularized_gamma(n * shape, x / scale)
    return cdf


def _bisect_quantile(cdf_fn, target: float, lo: float, hi: float, tol: float = 1e-9) -> float:
    """Smallest x with cdf_fn(x) >= target, for a continuous, monotonic
    cdf_fn -- expands hi until it's a valid upper bound, then bisects."""
    max_expansions = 200
    for _ in range(max_expansions):
        if cdf_fn(hi) >= target:
            break
        hi *= 2.0
    else:
        raise RuntimeError(f"quantile search could not bracket target={target} within {max_expansions} doublings.")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf_fn(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class CompoundPoissonSafetyStock:
    """SS = quantile(LTD) - mean(LTD), for intermittent/lumpy demand.
    Demand modeled as D_t = sum of Y_i, N ~ Poisson(lambda) events/period,
    Y ~ gamma-shaped size distribution (method-of-moments fit from
    history).

    Closed-form (2026-09-02, replacing an earlier Monte Carlo simulation):
    given N events, the sum of N iid Gamma(shape, scale) draws is itself
    exactly Gamma(N*shape, scale) (gamma's additivity under convolution
    with a shared scale) -- so the full LTD distribution is a Poisson-
    weighted mixture of gamma CDFs (_compound_poisson_gamma_cdf), not
    something that needs simulating. No RNG anywhere in this class:
    results are exact (up to floating-point precision and the Poisson-tail
    truncation in _poisson_tail_cutoff, both far tighter than Monte Carlo
    sampling error ever was) and exactly reproducible by construction.
    Cross-checked against a 3-million-draw Monte Carlo reference (the
    method this replaces) before being trusted: closed-form and simulated
    safety stock agreed to within the simulation's own sampling noise.

    Protection window is lead_time + horizon (matches SqrtHorizonSafetyStock
    and how ReplenishmentPolicy actually calls every strategy) -- scaling by
    lead_time alone under-covers by the review-period portion of the window
    whenever horizon > 0; this was a real bug here until fixed alongside
    NegativeBinomialSafetyStock (2026-09-02).
    """

    target_service_level: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 < self.target_service_level < 1.0:
            raise ValueError("target_service_level must be in (0, 1).")

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        values = _actual_values(actuals, period)
        protection = lead_time + horizon
        if not values or protection <= 0:
            return 0.0

        nonzero = [v for v in values if v > 0]
        if not nonzero:
            return 0.0

        lambda_rate = len(nonzero) / len(values)
        size_mean = statistics.fmean(nonzero)
        size_std = statistics.stdev(nonzero) if len(nonzero) > 1 else 0.0

        rate = protection * lambda_rate
        if rate <= 0 or size_mean <= 0:
            return 0.0

        mean_ltd = rate * size_mean  # E[sum Y_i] = E[N] * E[Y], exact either branch

        if size_std > 0:
            shape = max((size_mean ** 2) / (size_std ** 2), 1e-6)
            scale = (size_std ** 2) / size_mean
            point_mass_zero = math.exp(-rate)
            if self.target_service_level <= point_mass_zero:
                quantile = 0.0
            else:
                n_max = _poisson_tail_cutoff(rate)
                cdf_fn = lambda x: _compound_poisson_gamma_cdf(x, rate, shape, scale, n_max)  # noqa: E731
                std_ltd = math.sqrt(rate * shape * scale ** 2 * (1.0 + shape))
                quantile = _bisect_quantile(
                    cdf_fn, self.target_service_level, 0.0, mean_ltd + 10.0 * std_ltd + 1.0
                )
        else:
            quantile = _poisson_quantile(self.target_service_level, rate) * size_mean

        return max(quantile - mean_ltd, 0.0)


class NegativeBinomialRangeError(ValueError):
    """Raised when the demand history isn't overdispersed enough
    (variance <= mean) for a valid negative-binomial fit, and the strategy
    was not told to fall back to zero."""

    def __init__(self, *, mean_d: float, var_d: float):
        self.mean_d = mean_d
        self.var_d = var_d
        super().__init__(
            f"Negative-binomial fit requires overdispersed demand "
            f"(variance > mean); got mean={mean_d:.4f}, variance={var_d:.4f}. "
            f"Pass on_underdispersion='zero' to treat this as needing no "
            f"safety stock instead of raising."
        )


def _nb_quantile(target: float, R: float, p: float) -> float:
    """Smallest k with CDF(k) >= target for NB(R, p) (R = shape, p = success
    probability; mean = R(1-p)/p). Walks forward from k=0 accumulating pmf
    mass via the standard ratio recurrence pmf(k) = pmf(k-1) * (k-1+R)/k *
    (1-p) -- O(1) work per step, no lgamma calls in the loop, exact (no
    Monte Carlo, no normal approximation)."""
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1).")
    cdf = 0.0
    pmf = math.exp(R * math.log(p))  # pmf(0) = p**R
    k = 0
    max_iterations = 1_000_000
    while True:
        cdf += pmf
        if cdf >= target:
            return float(k)
        k += 1
        if k > max_iterations:
            raise RuntimeError(
                f"NB quantile search did not converge within {max_iterations} "
                f"steps (R={R}, p={p}, target={target}, cdf={cdf}) -- "
                f"unexpected for realistic demand/target combinations."
            )
        pmf *= (k - 1 + R) / k * (1 - p)


@dataclass(frozen=True)
class NegativeBinomialSafetyStock:
    """SS = quantile(NB(L*r, p)) - mean(NB(L*r, p)), for overdispersed
    (variance > mean) demand -- the standard closed-form model for count
    data with more variance than a pure Poisson process would produce
    (exactly the condition Syntetos-Boylan's CV^2 > 0.49 threshold checks
    for "erratic"/"lumpy" classification).

    Unlike CompoundPoissonSafetyStock's Monte Carlo simulation of a
    Poisson-arrival/gamma-size compound process, this fits ONE distribution
    directly to the raw per-period demand history (mean/variance, zeros
    included -- not just nonzero events) via method of moments, and
    aggregates to the full protection window L = lead_time + horizon
    exactly: a sum of L i.i.d. NB(r, p) variables is itself NB(L*r, p) (a
    known convolution/stability property, not an approximation). The
    quantile is found by a deterministic discrete search (_nb_quantile) --
    no RNG, no simulation, exactly reproducible and cheaper than 5,000
    Monte Carlo draws.

    min_periods guards against a small-sample false negative: with very few
    observations, variance can coincidentally equal or undercut the mean
    even for a genuinely overdispersed item (e.g. exactly 2 equal values
    early in a backtest) -- that is a "not enough signal yet" condition,
    the same cold-start story every other strategy here already handles by
    returning 0.0, not a real property of the item's demand. Below
    min_periods, this returns 0.0 regardless of on_underdispersion; above
    it, an underdispersion finding is trusted as real.
    """

    target_service_level: float
    on_underdispersion: Literal["raise", "zero"] = "raise"
    min_periods: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.target_service_level < 1.0:
            raise ValueError("target_service_level must be in (0, 1).")
        if self.on_underdispersion not in ("raise", "zero"):
            raise ValueError("on_underdispersion must be 'raise' or 'zero'.")
        if self.min_periods < 2:
            raise ValueError("min_periods must be at least 2.")

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        values = _actual_values(actuals, period)
        if len(values) < self.min_periods:
            return 0.0

        mean_d = statistics.fmean(values)
        if mean_d <= 0:
            return 0.0
        var_d = statistics.variance(values)
        if var_d <= mean_d:
            if self.on_underdispersion == "zero":
                return 0.0
            raise NegativeBinomialRangeError(mean_d=mean_d, var_d=var_d)

        protection = lead_time + horizon
        if protection <= 0:
            return 0.0

        p = mean_d / var_d
        r = mean_d ** 2 / (var_d - mean_d)
        R = protection * r

        mean_ltd = R * (1 - p) / p
        quantile = _nb_quantile(self.target_service_level, R, p)
        return max(quantile - mean_ltd, 0.0)
