"""Distributional safety-stock strategies: computed from the raw demand
distribution (and, for King's formula, lead-time distribution), not from
forecast error. Ported from supplyplanning/prototype/engine
(kings_safety_stock, compound_poisson.py).

Kept separate from safety_stock.py because these don't use
_error_series/_require_actuals -- they read actuals directly, and
KingsFormulaSafetyStock works with no forecast error history at all
(period 0 -> zero std, SS=0, same cold-start behavior as the error-based
strategies).
"""
from __future__ import annotations

import math
import random
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


def _poisson(rate: float, rng: random.Random) -> int:
    """Knuth's algorithm. ponytail: O(rate) per draw, fine at the
    n_simulations=1000-ish scale this strategy defaults to; swap for an
    inversion/PTRS sampler if rate or n_simulations grows large."""
    if rate <= 0:
        return 0
    limit = math.exp(-rate)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= limit:
            return k - 1


@dataclass(frozen=True)
class CompoundPoissonSafetyStock:
    """SS = quantile(simulated LTD) - mean(simulated LTD), for intermittent
    / lumpy demand. Demand modeled as D_t = sum of Y_i, N ~ Poisson(lambda)
    events/period, Y ~ gamma-shaped size distribution (method-of-moments
    fit from history). Simulation-based (transparent, traceable) rather
    than a saddle-point approximation.

    Protection window is lead_time + horizon (matches SqrtHorizonSafetyStock
    and how ReplenishmentPolicy actually calls every strategy) -- scaling by
    lead_time alone under-covers by the review-period portion of the window
    whenever horizon > 0; this was a real bug here until fixed alongside
    NegativeBinomialSafetyStock (2026-09-02).

    seed defaults to 0, not None: a shared-library default that draws fresh
    OS entropy on every call makes every caller's results non-reproducible
    by default, which is the wrong default for a policy-simulation library
    where "rerun the same backtest, get the same numbers" is an expected
    property. Pass seed=None explicitly for fresh randomness each call.

    n_simulations defaults to 1000, not 5000: Monte Carlo quantile error
    scales as sqrt(p(1-p)/n), so 1000 vs 5000 only widens the standard
    error on a 95th-percentile estimate from ~0.3% to ~0.7% of rank
    position -- small next to everything else already approximate in a
    demand-planning pipeline (the fitted demand model, tier-pooling, etc.),
    while this method is typically called many thousands of times per
    backtest (every period x every item x every candidate k), where total
    runtime scales linearly with n_simulations. Raise it for a final,
    one-off high-precision confirmation run if needed.
    """

    target_service_level: float = 0.95
    n_simulations: int = 1000
    seed: int | None = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_service_level < 1.0:
            raise ValueError("target_service_level must be in (0, 1).")
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")

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

        rng = random.Random(self.seed)
        if size_std > 0:
            shape = max((size_mean ** 2) / (size_std ** 2), 1e-6)
            scale = (size_std ** 2) / size_mean
            ltds = []
            for _ in range(self.n_simulations):
                n_events = _poisson(rate, rng)
                ltds.append(sum(rng.gammavariate(shape, scale) for _ in range(n_events)) if n_events else 0.0)
        else:
            ltds = [_poisson(rate, rng) * size_mean for _ in range(self.n_simulations)]

        ltds.sort()
        idx = min(int(self.target_service_level * len(ltds)), len(ltds) - 1)
        quantile_value = ltds[idx]
        mean_ltd = statistics.fmean(ltds)
        return max(quantile_value - mean_ltd, 0.0)


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
