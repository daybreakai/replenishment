import math
import statistics

import pytest

from replenishment.math_ import normal_quantile
from replenishment.timeseries import TimeSeries
from replenishment.strategies.distributional_safety_stock import (
    KingsFormulaSafetyStock, CompoundPoissonSafetyStock,
    NegativeBinomialSafetyStock, NegativeBinomialRangeError, _nb_quantile,
)

SMOOTH_ACTUALS = TimeSeries.from_values([10, 12, 8, 11, 9, 10, 13, 7, 10, 10])
FORECAST = TimeSeries.from_values([10] * 10)


def test_kings_formula_matches_hand_computed_full_protection_window():
    """Protection window is lead_time + horizon (2026-09-02 fix) -- scaling
    by lead_time alone under-covers by the review-period portion whenever
    horizon > 0, which it always is when called through ReplenishmentPolicy
    (ReplenishmentPolicy.order_quantity_for passes forecast_horizon, never
    0)."""
    strategy = KingsFormulaSafetyStock(target_service_level=0.95)
    period, lead_time, horizon = 8, 3, 2
    values = [10, 12, 8, 11, 9, 10, 13, 7]
    mean_d, std_d = statistics.fmean(values), statistics.stdev(values)
    expected = normal_quantile(0.95) * math.sqrt((lead_time + horizon) * std_d ** 2)

    ss = strategy.compute(
        forecast=FORECAST, actuals=SMOOTH_ACTUALS, period=period,
        lead_time=lead_time, horizon=horizon, service_level_factor=0.95,
    )
    assert abs(ss - expected) < 1e-9


def test_kings_formula_buffer_changes_with_horizon():
    """Regression guard: horizon must actually affect the buffer (it silently
    didn't, before the 2026-09-02 fix)."""
    strategy = KingsFormulaSafetyStock(target_service_level=0.95)
    ss_h1 = strategy.compute(forecast=FORECAST, actuals=SMOOTH_ACTUALS, period=8,
                              lead_time=3, horizon=1, service_level_factor=0.95)
    ss_h3 = strategy.compute(forecast=FORECAST, actuals=SMOOTH_ACTUALS, period=8,
                              lead_time=3, horizon=3, service_level_factor=0.95)
    assert ss_h3 > ss_h1


def test_kings_formula_includes_lead_time_variance_term_when_given():
    strategy = KingsFormulaSafetyStock(target_service_level=0.95, std_lead_time_periods=1.5)
    ss_with_lt_var = strategy.compute(
        forecast=FORECAST, actuals=SMOOTH_ACTUALS, period=8,
        lead_time=3, horizon=1, service_level_factor=0.95,
    )
    ss_without = KingsFormulaSafetyStock(target_service_level=0.95).compute(
        forecast=FORECAST, actuals=SMOOTH_ACTUALS, period=8,
        lead_time=3, horizon=1, service_level_factor=0.95,
    )
    assert ss_with_lt_var > ss_without


def test_kings_formula_zero_at_period_zero_no_history():
    strategy = KingsFormulaSafetyStock()
    ss = strategy.compute(forecast=FORECAST, actuals=SMOOTH_ACTUALS, period=0, lead_time=3, horizon=1, service_level_factor=0.95)
    assert ss == 0.0


def test_kings_formula_requires_actuals():
    strategy = KingsFormulaSafetyStock()
    with pytest.raises(ValueError, match="actuals"):
        strategy.compute(forecast=FORECAST, actuals=None, period=8, lead_time=3, horizon=1, service_level_factor=0.95)


def test_kings_formula_rejects_target_outside_open_interval():
    with pytest.raises(ValueError):
        KingsFormulaSafetyStock(target_service_level=1.0)


def test_compound_poisson_positive_for_lumpy_demand():
    lumpy_actuals = TimeSeries.from_values([0, 0, 0, 5, 0, 0, 0, 80, 0, 0, 0, 3, 0, 0])
    strategy = CompoundPoissonSafetyStock(target_service_level=0.95, n_simulations=2000, seed=0)
    ss = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=14, lead_time=3, horizon=1, service_level_factor=0.95)
    assert ss > 0


def test_compound_poisson_zero_protection_window_is_zero():
    """Protection is lead_time + horizon (2026-09-02 fix) -- both must be
    zero for a truly zero protection window. lead_time=0 alone no longer
    short-circuits to zero (see next test): it was doing so incorrectly
    before the fix, ignoring a nonzero horizon's review-period demand."""
    lumpy_actuals = TimeSeries.from_values([0, 0, 0, 5, 0, 0, 0, 80])
    strategy = CompoundPoissonSafetyStock(seed=0)
    ss = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=8, lead_time=0, horizon=0, service_level_factor=0.95)
    assert ss == 0.0


def test_compound_poisson_zero_lead_time_still_protects_against_horizon():
    """Regression guard for the 2026-09-02 fix: lead_time=0 with a nonzero
    horizon must still produce a buffer (it incorrectly returned 0.0 before,
    ignoring the review-period portion of the protection window)."""
    lumpy_actuals = TimeSeries.from_values([0, 0, 0, 5, 0, 0, 0, 80])
    strategy = CompoundPoissonSafetyStock(seed=0)
    ss = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=8, lead_time=0, horizon=1, service_level_factor=0.95)
    assert ss > 0.0


def test_compound_poisson_all_zero_history_is_zero():
    zero_actuals = TimeSeries.from_values([0] * 10)
    strategy = CompoundPoissonSafetyStock(seed=0)
    ss = strategy.compute(forecast=FORECAST, actuals=zero_actuals, period=10, lead_time=3, horizon=1, service_level_factor=0.95)
    assert ss == 0.0


def test_compound_poisson_is_deterministic_given_seed():
    lumpy_actuals = TimeSeries.from_values([0, 0, 0, 5, 0, 0, 0, 80, 0, 0, 0, 3])
    strategy = CompoundPoissonSafetyStock(n_simulations=1000, seed=42)
    ss1 = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=12, lead_time=2, horizon=1, service_level_factor=0.95)
    ss2 = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=12, lead_time=2, horizon=1, service_level_factor=0.95)
    assert ss1 == ss2


def test_compound_poisson_deterministic_by_default_without_explicit_seed():
    """2026-09-02: seed now defaults to 0, not None -- rerunning the same
    backtest must give the same numbers without the caller opting in."""
    lumpy_actuals = TimeSeries.from_values([0, 0, 0, 5, 0, 0, 0, 80, 0, 0, 0, 3])
    strategy = CompoundPoissonSafetyStock(n_simulations=500)
    ss1 = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=12, lead_time=2, horizon=1, service_level_factor=0.95)
    ss2 = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=12, lead_time=2, horizon=1, service_level_factor=0.95)
    assert ss1 == ss2


def test_compound_poisson_buffer_changes_with_horizon():
    """Regression guard for the 2026-09-02 fix: horizon must affect the
    buffer (it silently didn't before)."""
    lumpy_actuals = TimeSeries.from_values([0, 0, 0, 5, 0, 0, 0, 80, 0, 0, 0, 3, 0, 0])
    strategy = CompoundPoissonSafetyStock(n_simulations=2000, seed=0)
    ss_h1 = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=14, lead_time=3, horizon=1, service_level_factor=0.95)
    ss_h4 = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=14, lead_time=3, horizon=4, service_level_factor=0.95)
    assert ss_h4 > ss_h1


NB_VALUES = [0, 0, 0, 1, 2, 3, 3, 4, 20, 25, 0, 1, 0, 2, 3, 0, 1, 0, 0, 4,
             0, 0, 1, 2, 0, 0, 3, 20, 0, 1] * 2  # overdispersed by construction
NB_ACTUALS = TimeSeries.from_values(NB_VALUES)


def test_nb_quantile_matches_brute_force_cdf_summation():
    """_nb_quantile's O(1)-per-step ratio recurrence must agree with a
    direct (independent) CDF summation via the log-pmf formula."""
    for target, R, p in [(0.95, 5.0, 0.3), (0.90, 2.0, 0.1), (0.99, 50.0, 0.6), (0.80, 1.0, 0.05)]:
        cdf, k = 0.0, 0
        while True:
            log_pmf = math.lgamma(k + R) - math.lgamma(R) - math.lgamma(k + 1) + R * math.log(p) + k * math.log(1 - p)
            cdf += math.exp(log_pmf)
            if cdf >= target:
                break
            k += 1
        assert _nb_quantile(target, R, p) == k


def test_negative_binomial_matches_hand_computed_fit_and_quantile():
    values = NB_VALUES
    mean_d, var_d = statistics.fmean(values), statistics.variance(values)
    assert var_d > mean_d  # precondition for this test to exercise the real path

    lead_time, horizon = 2, 3
    p = mean_d / var_d
    r = mean_d ** 2 / (var_d - mean_d)
    R = (lead_time + horizon) * r
    expected_mean_ltd = R * (1 - p) / p
    expected_quantile = _nb_quantile(0.95, R, p)
    expected_ss = max(expected_quantile - expected_mean_ltd, 0.0)

    strategy = NegativeBinomialSafetyStock(target_service_level=0.95)
    ss = strategy.compute(forecast=FORECAST, actuals=NB_ACTUALS, period=len(values),
                          lead_time=lead_time, horizon=horizon, service_level_factor=1.0)
    assert abs(ss - expected_ss) < 1e-9


def test_negative_binomial_is_deterministic():
    """No RNG anywhere in this strategy -- repeated calls must be
    byte-identical, unlike CompoundPoissonSafetyStock pre-seed-fix."""
    strategy = NegativeBinomialSafetyStock(target_service_level=0.95)
    values_len = len(NB_VALUES)
    ss1 = strategy.compute(forecast=FORECAST, actuals=NB_ACTUALS, period=values_len,
                          lead_time=2, horizon=3, service_level_factor=1.0)
    ss2 = strategy.compute(forecast=FORECAST, actuals=NB_ACTUALS, period=values_len,
                          lead_time=2, horizon=3, service_level_factor=1.0)
    assert ss1 == ss2


def test_negative_binomial_buffer_changes_with_horizon():
    strategy = NegativeBinomialSafetyStock(target_service_level=0.95)
    values_len = len(NB_VALUES)
    ss_h1 = strategy.compute(forecast=FORECAST, actuals=NB_ACTUALS, period=values_len,
                            lead_time=2, horizon=1, service_level_factor=1.0)
    ss_h5 = strategy.compute(forecast=FORECAST, actuals=NB_ACTUALS, period=values_len,
                            lead_time=2, horizon=5, service_level_factor=1.0)
    assert ss_h5 > ss_h1


def test_negative_binomial_underdispersion_raises_by_default():
    smooth = TimeSeries.from_values([10] * 30)  # zero variance -> underdispersed
    strategy = NegativeBinomialSafetyStock(target_service_level=0.95)
    with pytest.raises(NegativeBinomialRangeError, match="overdispersed"):
        strategy.compute(forecast=FORECAST, actuals=smooth, period=30,
                        lead_time=2, horizon=3, service_level_factor=1.0)


def test_negative_binomial_underdispersion_zero_fallback():
    smooth = TimeSeries.from_values([10] * 30)
    strategy = NegativeBinomialSafetyStock(target_service_level=0.95, on_underdispersion="zero")
    ss = strategy.compute(forecast=FORECAST, actuals=smooth, period=30,
                         lead_time=2, horizon=3, service_level_factor=1.0)
    assert ss == 0.0


def test_negative_binomial_rejects_target_outside_open_interval():
    with pytest.raises(ValueError):
        NegativeBinomialSafetyStock(target_service_level=1.0)


def test_negative_binomial_rejects_bad_on_underdispersion():
    with pytest.raises(ValueError):
        NegativeBinomialSafetyStock(target_service_level=0.95, on_underdispersion="bogus")


def test_negative_binomial_small_sample_underdispersion_returns_zero_not_raise():
    """Regression guard: 2 equal observations trivially look underdispersed
    (variance=0) but that's a small-sample artifact, not a real property of
    the item -- must return 0.0 (cold-start behavior), never raise, below
    min_periods. This is the exact shape that crashed a real backtest run
    before this guard existed."""
    two_equal = TimeSeries.from_values([8, 8])
    strategy = NegativeBinomialSafetyStock(target_service_level=0.95)
    ss = strategy.compute(forecast=FORECAST, actuals=two_equal, period=2,
                         lead_time=1, horizon=2, service_level_factor=1.0)
    assert ss == 0.0


def test_negative_binomial_zero_at_period_zero_no_history():
    strategy = NegativeBinomialSafetyStock(target_service_level=0.95)
    ss = strategy.compute(forecast=FORECAST, actuals=NB_ACTUALS, period=0,
                         lead_time=2, horizon=3, service_level_factor=1.0)
    assert ss == 0.0


def test_negative_binomial_requires_actuals():
    strategy = NegativeBinomialSafetyStock(target_service_level=0.95)
    with pytest.raises(ValueError, match="actuals"):
        strategy.compute(forecast=FORECAST, actuals=None, period=8,
                         lead_time=2, horizon=3, service_level_factor=1.0)


def test_compound_poisson_rejects_zero_simulations():
    with pytest.raises(ValueError):
        CompoundPoissonSafetyStock(n_simulations=0)
