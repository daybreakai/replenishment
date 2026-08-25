import math
import statistics

import pytest

from replenishment.math_ import normal_quantile
from replenishment.timeseries import TimeSeries
from replenishment.strategies.distributional_safety_stock import (
    KingsFormulaSafetyStock, CompoundPoissonSafetyStock,
)

SMOOTH_ACTUALS = TimeSeries.from_values([10, 12, 8, 11, 9, 10, 13, 7, 10, 10])
FORECAST = TimeSeries.from_values([10] * 10)


def test_kings_formula_matches_hand_computed_deterministic_lead_time():
    strategy = KingsFormulaSafetyStock(target_service_level=0.95)
    period, lead_time = 8, 3
    values = [10, 12, 8, 11, 9, 10, 13, 7]
    mean_d, std_d = statistics.fmean(values), statistics.stdev(values)
    expected = normal_quantile(0.95) * math.sqrt(lead_time * std_d ** 2)

    ss = strategy.compute(
        forecast=FORECAST, actuals=SMOOTH_ACTUALS, period=period,
        lead_time=lead_time, horizon=1, service_level_factor=0.95,
    )
    assert abs(ss - expected) < 1e-9


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


def test_compound_poisson_zero_lead_time_is_zero():
    lumpy_actuals = TimeSeries.from_values([0, 0, 0, 5, 0, 0, 0, 80])
    strategy = CompoundPoissonSafetyStock(seed=0)
    ss = strategy.compute(forecast=FORECAST, actuals=lumpy_actuals, period=8, lead_time=0, horizon=1, service_level_factor=0.95)
    assert ss == 0.0


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


def test_compound_poisson_rejects_zero_simulations():
    with pytest.raises(ValueError):
        CompoundPoissonSafetyStock(n_simulations=0)
