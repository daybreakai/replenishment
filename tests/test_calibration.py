# tests/test_calibration.py
import pytest
from replenishment.calibration import optimize
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import SqrtHorizonSafetyStock
from replenishment.policy import ReplenishmentPolicy


def _policy_builder(factor: float) -> ReplenishmentPolicy:
    forecast = TimeSeries.from_values([10] * 60)
    actuals = TimeSeries.from_values([10] * 60)
    return ReplenishmentPolicy.order_up_to(
        forecast=forecast, actuals=actuals,
        safety_stock=SqrtHorizonSafetyStock(factor=factor),
        lead_time=2, forecast_horizon=1,
    )


def test_optimize_picks_a_candidate_from_the_provided_list():
    result = optimize(
        candidate_builder=_policy_builder, candidate_values=[0.5, 1.0, 1.65, 2.33],
        periods=40, demand=[10] * 40, initial_on_hand=20, lead_time=2,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0,
    )
    assert result.best_value in [0.5, 1.0, 1.65, 2.33]


def test_optimize_scores_on_a_held_out_slice_distinct_from_search_slice():
    # With validation_fraction=0.3 over 40 periods, the search slice is the
    # first 28 periods and the validation slice is the last 12 -- assert
    # the result records both slice sizes so the split is externally
    # verifiable, not an internal implementation detail nobody can check.
    result = optimize(
        candidate_builder=_policy_builder, candidate_values=[1.0, 1.65],
        periods=40, demand=[10] * 40, initial_on_hand=20, lead_time=2,
        holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0, validation_fraction=0.3,
    )
    assert result.search_periods == 28
    assert result.validation_periods == 12


def test_optimize_rejects_validation_slice_smaller_than_two_periods():
    with pytest.raises(ValueError, match="validation"):
        optimize(
            candidate_builder=_policy_builder, candidate_values=[1.0],
            periods=3, demand=[10, 10, 10], initial_on_hand=20, lead_time=0,
            holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0, validation_fraction=0.3,
        )


def test_optimize_rejects_empty_candidate_list():
    with pytest.raises(ValueError):
        optimize(
            candidate_builder=_policy_builder, candidate_values=[],
            periods=40, demand=[10] * 40, initial_on_hand=20, lead_time=2,
            holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0,
        )
