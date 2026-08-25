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
    """SS = z_alpha * sqrt(L * std_demand^2 + mean_demand^2 * std_lead_time^2).

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

        variance = lead_time * std_d ** 2 + mean_d ** 2 * self.std_lead_time_periods ** 2
        return z_alpha * math.sqrt(max(variance, 0.0))


def _poisson(rate: float, rng: random.Random) -> int:
    """Knuth's algorithm. ponytail: O(rate) per draw, fine at the
    n_simulations=5000-ish scale this strategy defaults to; swap for an
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
    """

    target_service_level: float = 0.95
    n_simulations: int = 5000
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.target_service_level < 1.0:
            raise ValueError("target_service_level must be in (0, 1).")
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        actuals = _require_actuals(actuals)
        values = _actual_values(actuals, period)
        if not values or lead_time <= 0:
            return 0.0

        nonzero = [v for v in values if v > 0]
        if not nonzero:
            return 0.0

        lambda_rate = len(nonzero) / len(values)
        size_mean = statistics.fmean(nonzero)
        size_std = statistics.stdev(nonzero) if len(nonzero) > 1 else 0.0

        rate = lead_time * lambda_rate
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
