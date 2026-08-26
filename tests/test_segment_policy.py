import pytest

from replenishment.segment_policy import (
    SegmentPolicyEntry,
    SegmentPolicyMap,
    load_segment_policy_map,
)


def test_resolve_falls_back_to_default_for_unknown_segment():
    policy = SegmentPolicyMap(default=SegmentPolicyEntry(factor=1.65, method="sqrt_horizon"))
    assert policy.resolve("nonexistent") == {"factor": 1.65, "method": "sqrt_horizon"}


def test_resolve_overrides_only_the_fields_a_segment_sets():
    policy = SegmentPolicyMap(
        default=SegmentPolicyEntry(factor=1.65, method="sqrt_horizon", mode="base_stock"),
        segments={"abc=A": SegmentPolicyEntry(factor=2.0)},
    )
    resolved = policy.resolve("abc=A")
    assert resolved == {"factor": 2.0, "method": "sqrt_horizon", "mode": "base_stock"}


def test_merge_runtime_override_wins_field_by_field():
    base = SegmentPolicyMap(
        default=SegmentPolicyEntry(factor=1.65, method="sqrt_horizon"),
        segments={"abc=A": SegmentPolicyEntry(factor=2.0, mode="base_stock")},
    )
    override = SegmentPolicyMap(segments={"abc=A": SegmentPolicyEntry(factor=3.0)})
    merged = base.merge(override)
    resolved = merged.resolve("abc=A")
    assert resolved["factor"] == 3.0
    assert resolved["mode"] == "base_stock"  # untouched by override
    assert resolved["method"] == "sqrt_horizon"  # inherited from default


def test_merge_adds_a_segment_only_present_in_override():
    base = SegmentPolicyMap(default=SegmentPolicyEntry(factor=1.65))
    override = SegmentPolicyMap(segments={"group=kit": SegmentPolicyEntry(mode="rop")})
    merged = base.merge(override)
    assert merged.resolve("group=kit") == {"factor": 1.65, "mode": "rop"}


def test_load_segment_policy_map_from_yaml(tmp_path):
    yaml_text = """
default:
  factor: 1.65
  method: sqrt_horizon
segments:
  "abc=A|demand_pattern=lumpy":
    factor: 2.0
    method: k_rmse
"""
    path = tmp_path / "policy.yaml"
    path.write_text(yaml_text)
    policy = load_segment_policy_map(path)
    assert policy.resolve("abc=A|demand_pattern=lumpy") == {"factor": 2.0, "method": "k_rmse"}
    assert policy.resolve("other") == {"factor": 1.65, "method": "sqrt_horizon"}


def test_entry_rejects_unknown_field():
    with pytest.raises(Exception):
        SegmentPolicyEntry(not_a_real_knob=1)
