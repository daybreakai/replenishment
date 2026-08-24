"""Day-by-day inventory simulation. Ported from janrth's simulation.py —
unchanged except it now calls a single ReplenishmentPolicy type instead
of any of 7 OrderingPolicy-protocol implementers.

Note: this is a lost-sales model, not a backorder model. Unmet demand in a
period is recorded (via InventorySnapshot.backorders / SimulationSummary
stockout accounting) but never carried forward into future on_hand or
inventory_position — it is not backfilled once new stock arrives.

The period_offset parameter allows a simulation to run over a slice of a
longer timeline while ensuring the policy's forecast/actuals read at the
correct absolute position. For example, calibration.optimize uses it to
make validation runs genuinely out-of-sample: search runs at offset=0,
validation runs at offset=search_periods, so both access the policy's
forecast/actuals at the right absolute indices rather than restarting
from the beginning.

Similarly, initial_pipeline/ending_pipeline let a caller run a simulation
over a slice of a longer timeline while in-transit orders from the
previous slice continue to arrive on schedule, rather than the pipeline
restarting empty. Pass the previous run's ending_pipeline as this run's
initial_pipeline to carry them forward.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

DemandModel = Callable[[int], int]


@dataclass(frozen=True)
class InventoryState:
    period: int
    on_hand: int
    on_order: int
    backorders: int

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


@dataclass(frozen=True)
class InventorySnapshot:
    period: int
    starting_on_hand: int
    demand: int
    received: int
    ending_on_hand: int
    backorders: int
    order_placed: int
    on_order: int


@dataclass(frozen=True)
class SimulationSummary:
    total_demand: int
    total_fulfilled: int
    total_backorders: int
    fill_rate: float
    average_on_hand: float
    holding_cost: float
    stockout_cost: float
    ordering_cost: float
    total_cost: float


@dataclass(frozen=True)
class SimulationResult:
    snapshots: list[InventorySnapshot]
    summary: SimulationSummary
    ending_pipeline: list[int]


def _normalize_demand(demand: Iterable[int] | DemandModel) -> DemandModel:
    if callable(demand):
        return demand
    demand_list = list(demand)

    def demand_model(period: int) -> int:
        if period < 0 or period >= len(demand_list):
            raise IndexError("Demand period out of range.")
        return demand_list[period]

    return demand_model


def simulate_replenishment(
    *, periods: int, demand: Iterable[int] | DemandModel, initial_on_hand: int,
    lead_time: int, policy, holding_cost_per_unit: float = 0.0,
    stockout_cost_per_unit: float = 0.0, order_cost_per_order: float = 0.0,
    order_cost_per_unit: float = 0.0, period_offset: int = 0,
    initial_pipeline: list[int] | None = None,
) -> SimulationResult:
    if periods <= 0:
        raise ValueError("periods must be positive.")
    if lead_time < 0:
        raise ValueError("lead_time cannot be negative.")

    if callable(demand):
        demand_model = demand
    else:
        demand = list(demand)  # materialize once; safe to re-list() a list below
        if periods > len(demand):
            raise ValueError(
                f"periods ({periods}) exceeds the length of the provided demand series ({len(demand)})."
            )
        demand_model = _normalize_demand(demand)
    on_hand = initial_on_hand
    if initial_pipeline is None:
        pipeline: list[int] = [0 for _ in range(lead_time)]
    else:
        if len(initial_pipeline) != lead_time:
            raise ValueError(
                f"initial_pipeline must have length lead_time ({lead_time}), got {len(initial_pipeline)}."
            )
        pipeline = list(initial_pipeline)
    snapshots: list[InventorySnapshot] = []

    total_demand = 0
    total_fulfilled = 0
    total_backorders = 0
    on_hand_total = 0
    ordering_cost_total = 0.0

    for period in range(periods):
        absolute_period = period + period_offset
        received = pipeline.pop(0) if lead_time > 0 else 0
        on_hand += received
        period_demand = demand_model(period)
        total_demand += period_demand

        fulfilled = min(on_hand, period_demand)
        on_hand -= fulfilled
        unmet = period_demand - fulfilled
        total_backorders += unmet

        state = InventoryState(period=absolute_period, on_hand=on_hand, on_order=sum(pipeline), backorders=0)
        order_qty = max(0, policy.order_quantity_for(state))
        if order_qty > 0:
            ordering_cost_total += order_cost_per_order + (order_cost_per_unit * order_qty)
        if lead_time == 0:
            on_hand += order_qty
        else:
            pipeline.append(order_qty)

        total_fulfilled += fulfilled
        on_hand_total += on_hand
        snapshots.append(InventorySnapshot(
            period=absolute_period, starting_on_hand=on_hand + fulfilled, demand=period_demand,
            received=received, ending_on_hand=on_hand, backorders=unmet,
            order_placed=order_qty, on_order=sum(pipeline),
        ))

    fill_rate = total_fulfilled / total_demand if total_demand else 1.0
    average_on_hand = on_hand_total / periods
    holding_cost = on_hand_total * holding_cost_per_unit
    stockout_cost = total_backorders * stockout_cost_per_unit
    total_cost = holding_cost + stockout_cost + ordering_cost_total

    summary = SimulationSummary(
        total_demand=total_demand, total_fulfilled=total_fulfilled,
        total_backorders=total_backorders, fill_rate=fill_rate,
        average_on_hand=average_on_hand, holding_cost=holding_cost,
        stockout_cost=stockout_cost, ordering_cost=ordering_cost_total, total_cost=total_cost,
    )
    return SimulationResult(snapshots=snapshots, summary=summary, ending_pipeline=list(pipeline))
