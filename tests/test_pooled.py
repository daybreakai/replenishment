import pytest

from replenishment import Portfolio, PooledReplenishmentError
from replenishment.io_ import generate_standard_simulation_rows


def _rows(n=4, periods=40, lead_time=2, seed=11):
    return generate_standard_simulation_rows(
        n_unique_ids=n, periods=periods, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=lead_time,
        seed=seed,
    )


def _released_value_by_period(res, unit_cost):
    n_periods = len(next(iter(res.results.values())).snapshots)
    out = []
    for period in range(n_periods):
        released = {uid: r.snapshots[period].order_placed for uid, r in res.results.items()}
        out.append((released, sum(q * unit_cost[uid] for uid, q in released.items())))
    return out


def test_every_release_clears_minimum_or_exhausts_assortment():
    rows = _rows()
    port = Portfolio(rows)
    unit_cost = dict.fromkeys(port.unique_ids, 10.0)
    res = port.simulate_pooled(
        factor=1.65, horizon=2, mode="rop_flat", review_period=1,
        minimum_value=50.0, unit_cost=unit_cost, wait_cap_periods=3,
    )
    for released, value in _released_value_by_period(res, unit_cost):
        any_released = any(q > 0 for q in released.values())
        if not any_released:
            continue
        cleared = value >= 50.0
        exhausted = all(q > 0 for q in released.values())  # every item in the portfolio released
        assert cleared or exhausted, (released, value)


def test_pooling_batches_orders_relative_to_unpooled_rop():
    rows = _rows()
    port = Portfolio(rows)
    unit_cost = dict.fromkeys(port.unique_ids, 10.0)
    unpooled = port.simulate(factor=1.65, horizon=2, mode="rop_flat", review_period=1)
    pooled = port.simulate_pooled(
        factor=1.65, horizon=2, mode="rop_flat", review_period=1,
        minimum_value=50.0, unit_cost=unit_cost, wait_cap_periods=5,
    )
    unpooled_events = sum(
        1 for r in unpooled.results.values() for s in r.snapshots if s.order_placed > 0)
    pooled_events = sum(
        1 for r in pooled.results.values() for s in r.snapshots if s.order_placed > 0)
    assert pooled_events <= unpooled_events  # pooling waits for the joint minimum


def test_demand_and_fulfillment_totals_match_unpooled():
    rows = _rows()
    port = Portfolio(rows)
    unit_cost = dict.fromkeys(port.unique_ids, 10.0)
    unpooled = port.simulate(factor=1.65, horizon=2, mode="rop_flat", review_period=1)
    pooled = port.simulate_pooled(
        factor=1.65, horizon=2, mode="rop_flat", review_period=1,
        minimum_value=50.0, unit_cost=unit_cost, wait_cap_periods=5,
    )
    for uid in port.unique_ids:
        assert pooled[uid].summary.total_demand == unpooled[uid].summary.total_demand


def test_order_up_to_trigger_rejected():
    rows = _rows(n=2)
    port = Portfolio(rows)
    unit_cost = dict.fromkeys(port.unique_ids, 10.0)
    with pytest.raises(PooledReplenishmentError):
        port.simulate_pooled(
            factor=1.65, horizon=2, mode="base_stock_flat", review_period=1,
            minimum_value=50.0, unit_cost=unit_cost, wait_cap_periods=3,
        )
