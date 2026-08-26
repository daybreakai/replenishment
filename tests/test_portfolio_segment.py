import pytest

from replenishment import Portfolio
from replenishment.io_ import StandardSimulationRow
from replenishment.segment_policy import SegmentPolicyEntry, SegmentPolicyMap
from replenishment.segmentation import ExplicitGroupRule


def _row(uid, ds, demand):
    return StandardSimulationRow(
        unique_id=uid, ds=ds, demand=demand, forecast=demand,
        actuals=demand, holding_cost_per_unit=1.0,
        stockout_cost_per_unit=5.0, order_cost_per_order=2.0,
        lead_time=2, initial_on_hand=10, current_stock=10,
        forecast_percentiles={},
    )


def _rows_for(uid, demands):
    return [_row(uid, f"2024-{i + 1:02d}-01", d) for i, d in enumerate(demands)]


def _portfolio():
    rows = _rows_for("hi", [10] * 6) + _rows_for("lo", [10] * 6)
    return Portfolio(rows)


def test_segment_delegates_to_segment_portfolio():
    port = _portfolio()
    rule = ExplicitGroupRule(groups={"hi": "g1"})
    keys = port.segment([rule])
    assert keys["hi"].labels == {"group": "g1"}
    assert keys["lo"].labels == {"group": "lo"}


def test_simulate_by_segment_applies_each_segments_own_knobs():
    port = _portfolio()
    rule = ExplicitGroupRule(groups={"hi": "g1"})
    policy = SegmentPolicyMap(
        default=SegmentPolicyEntry(factor=1.0, horizon=2),
        segments={"group=g1": SegmentPolicyEntry(factor=2.0, horizon=2)},
    )
    res = port.simulate_by_segment([rule], policy)

    expected_hi = Portfolio(_rows_for("hi", [10] * 6)).simulate(factor=2.0, horizon=2)
    expected_lo = Portfolio(_rows_for("lo", [10] * 6)).simulate(factor=1.0, horizon=2)
    assert res["hi"].summary == expected_hi["hi"].summary
    assert res["lo"].summary == expected_lo["lo"].summary


def test_simulate_by_segment_merges_runtime_overrides_first():
    port = _portfolio()
    rule = ExplicitGroupRule(groups={"hi": "g1"})
    policy = SegmentPolicyMap(default=SegmentPolicyEntry(factor=1.0, horizon=2))
    overrides = SegmentPolicyMap(segments={"group=g1": SegmentPolicyEntry(factor=3.0)})
    res = port.simulate_by_segment([rule], policy, overrides=overrides)

    expected_hi = Portfolio(_rows_for("hi", [10] * 6)).simulate(factor=3.0, horizon=2)
    expected_lo = Portfolio(_rows_for("lo", [10] * 6)).simulate(factor=1.0, horizon=2)
    assert res["hi"].summary == expected_hi["hi"].summary
    assert res["lo"].summary == expected_lo["lo"].summary


def test_simulate_by_segment_fails_loud_when_no_factor_resolvable():
    port = _portfolio()
    rule = ExplicitGroupRule(groups={"hi": "g1"})
    policy = SegmentPolicyMap(default=SegmentPolicyEntry(horizon=2))
    with pytest.raises(ValueError, match="group=g1"):
        port.simulate_by_segment([rule], policy)
