import pytest

from replenishment.io_ import StandardSimulationRow
from replenishment.segmentation import (
    ABCRule,
    AttributeRule,
    DemandPatternRule,
    ExplicitGroupRule,
    SegmentKey,
    XYZRule,
    segment_portfolio,
)


def _row(uid, ds, demand, holding_cost=1.0):
    return StandardSimulationRow(
        unique_id=uid, ds=ds, demand=demand, forecast=demand,
        actuals=demand, holding_cost_per_unit=holding_cost,
        stockout_cost_per_unit=3.0, order_cost_per_order=5.0,
        lead_time=1, initial_on_hand=0, current_stock=0,
        forecast_percentiles={},
    )


def _rows_for(uid, demands, holding_cost=1.0):
    return [_row(uid, f"2024-{i + 1:02d}-01", d, holding_cost) for i, d in enumerate(demands)]


def test_segment_key_id_is_sorted_and_deterministic():
    key = SegmentKey(labels={"xyz": "X", "abc": "A"})
    assert key.id == "abc=A|xyz=X"


def test_demand_pattern_rule_labels_by_classify_demand():
    rows = _rows_for("smooth-item", [10, 11, 9, 10, 10, 9, 11, 10])
    keys = segment_portfolio(rows, [DemandPatternRule()])
    assert keys["smooth-item"].labels == {"demand_pattern": "smooth"}


def test_abc_rule_ranks_across_items_by_value():
    rows = (
        _rows_for("big", [80, 80], holding_cost=5.0)
        + _rows_for("small", [5, 5], holding_cost=20.0)
    )
    keys = segment_portfolio(rows, [ABCRule()])
    assert keys["big"].labels["abc"] == "A"
    assert keys["small"].labels["abc"] == "C"


def test_xyz_rule_labels_steady_vs_erratic():
    rows = _rows_for("steady", [10, 10, 10, 10]) + _rows_for("erratic", [1, 50, 2, 45])
    keys = segment_portfolio(rows, [XYZRule()])
    assert keys["steady"].labels["xyz"] == "X"
    assert keys["erratic"].labels["xyz"] == "Z"


def test_attribute_rule_uses_supplied_tag_and_default():
    rows = _rows_for("tagged", [1, 2]) + _rows_for("untagged", [1, 2])
    rule = AttributeRule(attributes={"tagged": "supplier-x"})
    keys = segment_portfolio(rows, [rule])
    assert keys["tagged"].labels["attribute"] == "supplier-x"
    assert keys["untagged"].labels["attribute"] == "unassigned"


def test_explicit_group_rule_pins_a_shared_label_and_leaves_others_alone():
    rows = _rows_for("kit-a", [1, 2]) + _rows_for("kit-b", [1, 2]) + _rows_for("solo", [1, 2])
    rule = ExplicitGroupRule(groups={"kit-a": "kit", "kit-b": "kit"})
    keys = segment_portfolio(rows, [rule])
    assert keys["kit-a"].labels["group"] == "kit"
    assert keys["kit-b"].labels["group"] == "kit"
    assert keys["solo"].labels["group"] == "solo"


def test_composite_key_combines_every_rule():
    rows = _rows_for("item", [10, 11, 9, 10, 10, 9, 11, 10], holding_cost=1.0)
    keys = segment_portfolio(rows, [DemandPatternRule(), ABCRule(), XYZRule()])
    key = keys["item"]
    assert set(key.labels) == {"demand_pattern", "abc", "xyz"}
    assert key.id == f"abc={key.labels['abc']}|demand_pattern=smooth|xyz={key.labels['xyz']}"


def test_empty_portfolio_returns_empty():
    assert segment_portfolio([], [DemandPatternRule()]) == {}
