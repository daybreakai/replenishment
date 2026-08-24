"""Backtest + grid-search calibration, adapted from janrth's
optimize_service_level_factors et al. (optimization.py).

Fix vs. the original: candidates are scored on a held-out validation
slice, never on the same window used to search -- the original scored
and picked on one window, risking a factor fit to that window's noise
rather than the true cost tradeoff.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from replenishment.policy import ReplenishmentPolicy
from replenishment.simulation import simulate_replenishment


@dataclass(frozen=True)
class CalibrationResult:
    best_value: float
    best_validation_cost: float
    all_costs: dict[float, float]
    search_periods: int
    validation_periods: int


def optimize(
    *,
    candidate_builder: Callable[[float], ReplenishmentPolicy],
    candidate_values: Sequence[float],
    periods: int,
    demand: Sequence[int],
    initial_on_hand: int,
    lead_time: int,
    holding_cost_per_unit: float,
    stockout_cost_per_unit: float,
    order_cost_per_order: float = 0.0,
    validation_fraction: float = 0.3,
) -> CalibrationResult:
    if not candidate_values:
        raise ValueError("candidate_values must be non-empty.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")

    validation_periods = round(periods * validation_fraction)
    search_periods = periods - validation_periods
    if validation_periods < 2 or search_periods < 2:
        raise ValueError(
            f"validation slice ({validation_periods} periods) or search slice "
            f"({search_periods} periods) is too small to calibrate against "
            "meaningfully -- increase periods or adjust validation_fraction."
        )

    demand_list = list(demand)
    search_demand = demand_list[:search_periods]
    validation_demand = demand_list[search_periods:]

    all_costs: dict[float, float] = {}
    for value in candidate_values:
        policy = candidate_builder(value)
        search_result = simulate_replenishment(
            periods=search_periods, demand=search_demand, initial_on_hand=initial_on_hand,
            lead_time=lead_time, policy=policy, holding_cost_per_unit=holding_cost_per_unit,
            stockout_cost_per_unit=stockout_cost_per_unit, order_cost_per_order=order_cost_per_order,
        )
        # Carry ending on-hand AND the in-flight order pipeline from the
        # search window into the validation window so the two runs form one
        # continuous timeline -- otherwise orders still in transit at the
        # search/validation seam would be silently dropped. period_offset
        # ensures the validation policy reads forecast/actuals at the correct
        # absolute period (search_periods onward), not restarting at 0 -- this
        # is what makes validation scoring genuinely out-of-sample rather than
        # silently re-scoring against the start of the forecast history.
        ending_on_hand = search_result.snapshots[-1].ending_on_hand if search_result.snapshots else initial_on_hand
        validation_policy = candidate_builder(value)
        validation_result = simulate_replenishment(
            periods=validation_periods, demand=validation_demand, initial_on_hand=ending_on_hand,
            lead_time=lead_time, policy=validation_policy, holding_cost_per_unit=holding_cost_per_unit,
            stockout_cost_per_unit=stockout_cost_per_unit, order_cost_per_order=order_cost_per_order,
            period_offset=search_periods, initial_pipeline=search_result.ending_pipeline,
        )
        all_costs[value] = validation_result.summary.total_cost

    best_value = min(all_costs, key=lambda v: all_costs[v])
    return CalibrationResult(
        best_value=best_value, best_validation_cost=all_costs[best_value],
        all_costs=all_costs, search_periods=search_periods, validation_periods=validation_periods,
    )
