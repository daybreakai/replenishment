"""Joint/pooled replenishment: N items, each on its own continuous-review
reorder-point trigger, gated by one shared release event that must clear a
portfolio-wide $-value/weight minimum (e.g. a supplier's free-freight
threshold) before anything is actually placed.

Every item still gets its own on_hand trajectory -- demand and receipts are
per item, unaffected by pooling. Only the RELEASE of a triggered item's
order is gated: it waits in a shared pool with every other
currently-triggered item until the pool's total value (or weight) clears
the minimum, then everything in the pool releases together, in the same
period, as one consolidated order. This is what "enforce the minimum, order
whenever" means mechanically -- no fixed calendar cadence (each item
reviews every period), but nothing ships until the joint order clears the
supplier's threshold.

If no release has cleared the minimum within `wait_cap_periods` periods of
an item's oldest still-waiting trigger, the pool tops itself up from items
NOT yet at their own reorder point -- closest first (smallest
inventory_position - reorder_point gap) -- pulling each up to its own
order-up-to target, until the minimum clears or every item has been pulled
in. This bounds how long a genuinely short-on-stock item waits, without
padding the book by default (see MissouriTile's meeting-notes ruling,
2026-09-03: wait, don't pad, unless the wait itself is going too long).
"""
from __future__ import annotations

import math

from replenishment.io_ import ArticleSimulationConfig
from replenishment.policy import round_to_moq
from replenishment.simulation import InventorySnapshot, InventoryState, SimulationResult, SimulationSummary


class PooledReplenishmentError(Exception):
    """A pooled simulation couldn't be run. Always names the offending item/config."""


def simulate_pooled_replenishment(
    configs: dict[str, ArticleSimulationConfig],
    *,
    minimum_value: float,
    unit_cost: dict[str, float],
    wait_cap_periods: int,
    weight_by_item: dict[str, float] | None = None,
    minimum_weight: float | None = None,
    minimum_logic: str = "or",
) -> dict[str, SimulationResult]:
    if not configs:
        raise PooledReplenishmentError("simulate_pooled_replenishment needs at least one item")
    if minimum_logic not in ("or", "and"):
        raise PooledReplenishmentError(f"minimum_logic must be 'or' or 'and', got {minimum_logic!r}")
    if wait_cap_periods < 1:
        raise PooledReplenishmentError(f"wait_cap_periods must be >= 1, got {wait_cap_periods}")
    period_counts = {c.periods for c in configs.values()}
    if len(period_counts) != 1:
        raise PooledReplenishmentError(
            f"all items must share the same simulation length to pool orders — got "
            f"{sorted(period_counts)}")
    n_periods = period_counts.pop()
    no_status = sorted(uid for uid, c in configs.items() if not hasattr(c.policy.trigger, "status"))
    if no_status:
        raise PooledReplenishmentError(
            f"pooled simulation requires a reorder-point-family trigger (ReorderPointTrigger/"
            f"FlatReorderPointTrigger) for every item — missing .status() on: {no_status[:10]}")
    no_cost = sorted(uid for uid in configs if uid not in unit_cost)
    if no_cost:
        raise PooledReplenishmentError(f"unit_cost missing for: {no_cost[:10]}")

    on_hand = {uid: c.initial_on_hand for uid, c in configs.items()}
    pipeline = {uid: [0] * c.lead_time for uid, c in configs.items()}
    waiting_since: dict[str, int | None] = dict.fromkeys(configs, None)
    snapshots: dict[str, list[InventorySnapshot]] = {uid: [] for uid in configs}
    totals = {uid: {"demand": 0, "fulfilled": 0, "backorders": 0, "on_hand_total": 0,
                    "ordering_cost": 0.0} for uid in configs}

    for period in range(n_periods):
        received = {}
        for uid, c in configs.items():
            received[uid] = pipeline[uid].pop(0) if c.lead_time > 0 else 0
            on_hand[uid] += received[uid]
        starting_on_hand = dict(on_hand)  # post-receipt, pre-demand

        desired: dict[str, int] = {}
        states: dict[str, InventoryState] = {}
        for uid, c in configs.items():
            d = c.demand[period]
            totals[uid]["demand"] += d
            fulfilled = min(on_hand[uid], d)
            on_hand[uid] -= fulfilled
            unmet = d - fulfilled
            totals[uid]["fulfilled"] += fulfilled
            totals[uid]["backorders"] += unmet

            state = InventoryState(period=period, on_hand=on_hand[uid],
                                    on_order=sum(pipeline[uid]), backorders=0)
            states[uid] = state
            qty = c.policy.order_quantity_for(state)
            desired[uid] = qty
            if qty > 0:
                if waiting_since[uid] is None:
                    waiting_since[uid] = period
            else:
                waiting_since[uid] = None

        triggered = {uid for uid, qty in desired.items() if qty > 0}

        def _value(items: set) -> float:
            return sum(desired[u] * unit_cost[u] for u in items)

        def _weight(items: set) -> float:
            return sum(desired[u] * weight_by_item.get(u, 0.0) for u in items) if weight_by_item else 0.0

        def _clears(items: set) -> bool:
            if not items:
                return False
            usd_ok = _value(items) >= minimum_value
            lbs_ok = minimum_weight is not None and _weight(items) >= minimum_weight
            return (usd_ok or lbs_ok) if minimum_logic == "or" else (usd_ok and lbs_ok)

        release: set[str] = set()
        if triggered and _clears(triggered):
            release = set(triggered)
        elif triggered:
            oldest_wait = period - min(waiting_since[u] for u in triggered)
            if oldest_wait >= wait_cap_periods:
                release = set(triggered)
                candidates = []
                for uid, c in configs.items():
                    if uid in release:
                        continue
                    reorder_point, target = c.policy.status(states[uid])
                    gap = states[uid].inventory_position - reorder_point
                    candidates.append((gap, uid, target))
                candidates.sort(key=lambda x: x[0])
                for _gap, uid, target in candidates:
                    if _clears(release):
                        break
                    top_up = round_to_moq(
                        max(0, math.ceil(target - states[uid].inventory_position)),
                        configs[uid].policy.moq)
                    if top_up <= 0:
                        continue
                    desired[uid] = top_up
                    release.add(uid)
                # Still short even after pulling in the whole assortment: place
                # whatever the pool has rather than let triggered items starve
                # indefinitely -- the minimum genuinely can't be reached this period.

        for uid, c in configs.items():
            qty = desired[uid] if uid in release else 0
            if qty > 0:
                totals[uid]["ordering_cost"] += c.order_cost_per_order + c.order_cost_per_unit * qty
                waiting_since[uid] = None
            if c.lead_time == 0:
                on_hand[uid] += qty
            else:
                pipeline[uid].append(qty)
            totals[uid]["on_hand_total"] += on_hand[uid]
            snapshots[uid].append(InventorySnapshot(
                period=period, starting_on_hand=starting_on_hand[uid], demand=c.demand[period],
                received=received[uid], ending_on_hand=on_hand[uid],
                backorders=c.demand[period] - min(starting_on_hand[uid], c.demand[period]),
                order_placed=qty, on_order=sum(pipeline[uid]),
            ))

    results: dict[str, SimulationResult] = {}
    for uid, c in configs.items():
        t = totals[uid]
        fill_rate = t["fulfilled"] / t["demand"] if t["demand"] else 1.0
        average_on_hand = t["on_hand_total"] / n_periods
        holding_cost = t["on_hand_total"] * c.holding_cost_per_unit
        stockout_cost = t["backorders"] * c.stockout_cost_per_unit
        total_cost = holding_cost + stockout_cost + t["ordering_cost"]
        summary = SimulationSummary(
            total_demand=t["demand"], total_fulfilled=t["fulfilled"],
            total_backorders=t["backorders"], fill_rate=fill_rate,
            average_on_hand=average_on_hand, holding_cost=holding_cost,
            stockout_cost=stockout_cost, ordering_cost=t["ordering_cost"], total_cost=total_cost,
        )
        results[uid] = SimulationResult(
            snapshots=snapshots[uid], summary=summary, ending_pipeline=list(pipeline[uid]))
    return results
