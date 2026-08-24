"""Ports janrth's stress_test_example.ipynb scenario as a golden test.

Note on scope: stress_test_example.ipynb is a *performance* stress test --
it runs `optimize_aggregation_and_forecast_targets` over 50,000 synthetic
articles and reports only wall-clock time and the winning percentile
target ('p55'), not a total_cost or fill_rate. It also never calls
`random` -- demand is fully deterministic (`50 + idx % 10`). Additionally,
`optimize_aggregation_and_forecast_targets`/`ForecastCandidatesConfig`
(janrth's optimization.py/aggregation.py ecosystem) was explicitly not
ported into this repo (see io_.py's module docstring, Task 11) -- it's out
of scope for Tasks 1-12.

So this test cannot replay the notebook's own optimizer call, and there is
no notebook-reported total_cost/fill_rate number to check a tolerance
against. The closest faithful port is: take the notebook's actual demand
pattern and cost parameters (cell 4: periods=120, lead_time=1,
holding_cost_per_unit=0.8, stockout_cost_per_unit=3.5, initial_on_hand=40,
base_demand = 50 + idx % 10) and run them through this repo's ported
policy/simulation engine (SqrtHorizonSafetyStock + order-up-to +
simulate_replenishment) instead of the unported optimizer, then apply the
brief's fallback sanity bounds (total_cost > 0, fill_rate > 0.5) since
there's no exact figure to target.
"""
import pytest
from replenishment.simulation import simulate_replenishment
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import SqrtHorizonSafetyStock
from replenishment.policy import ReplenishmentPolicy

# Read from stress_test_example.ipynb cell 4 (article_count/periods/lead_time/
# holding_cost_per_unit/stockout_cost_per_unit/base_demand/initial_on_hand).
SCENARIO_PERIODS = 120
SCENARIO_LEAD_TIME = 1
SCENARIO_HOLDING_COST = 0.8
SCENARIO_STOCKOUT_COST = 3.5
SCENARIO_INITIAL_ON_HAND = 40
SCENARIO_BASE_DEMAND = [50 + (idx % 10) for idx in range(SCENARIO_PERIODS)]
SCENARIO_FORECAST_MEAN = sum(SCENARIO_BASE_DEMAND) / len(SCENARIO_BASE_DEMAND)


def test_golden_scenario_matches_pinned_regression_values():
    demand = SCENARIO_BASE_DEMAND
    forecast = TimeSeries.from_values([SCENARIO_FORECAST_MEAN] * SCENARIO_PERIODS)
    actuals = TimeSeries.from_values(demand)
    policy = ReplenishmentPolicy.order_up_to(
        forecast=forecast, actuals=actuals,
        safety_stock=SqrtHorizonSafetyStock(factor=1.65),
        lead_time=SCENARIO_LEAD_TIME, forecast_horizon=1,
    )
    result = simulate_replenishment(
        periods=SCENARIO_PERIODS, demand=demand, initial_on_hand=SCENARIO_INITIAL_ON_HAND,
        lead_time=SCENARIO_LEAD_TIME, policy=policy, holding_cost_per_unit=SCENARIO_HOLDING_COST,
        stockout_cost_per_unit=SCENARIO_STOCKOUT_COST, order_cost_per_order=0.0,
    )
    # The notebook never prints a total_cost/fill_rate to target exactly
    # (see module docstring), but this scenario is fully deterministic --
    # no random calls anywhere, demand is a fixed arithmetic sequence --
    # so unlike a stochastic scenario we CAN pin an exact regression value
    # once computed, rather than settling for loose sanity bounds. These
    # were computed by running this exact test as originally written and
    # observing the (stable, reproducible) result; if a future change to
    # the ported engine's math legitimately alters this number, update the
    # expected values here deliberately -- don't just loosen the tolerance.
    assert result.summary.total_cost == pytest.approx(737.4, abs=1e-6)
    assert result.summary.fill_rate == pytest.approx(0.9984709480122325, abs=1e-9)
