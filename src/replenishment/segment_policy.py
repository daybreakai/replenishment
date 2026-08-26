"""Segment -> policy-knob mapping, loaded from YAML and mergeable with a
runtime override map.

This is where a buying-rules conversation ends up: paste the buying rules
into chat, an engineer (or Claude) writes the committed default YAML,
Portfolio.simulate_by_segment resolves each item's SegmentKey.id against
it. No LLM in the simulation loop itself, this module only reads config.

    policy = load_segment_policy_map("policies/acme.yaml")
    policy.resolve("abc=A|demand_pattern=lumpy")
    # {"factor": 2.0, "method": "k_rmse", "mode": "base_stock"}
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class SegmentPolicyEntry(BaseModel):
    """One segment's override of the Portfolio.configs() knobs. Unset
    (None) fields mean "fall through to the map's default entry"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    factor: float | None = None
    horizon: int | None = None
    method: str | None = None
    mode: str | None = None
    review_period: int | None = None
    moq: int | None = None
    fixed_error: float | None = None

    def as_knobs(self) -> dict[str, float | int | str]:
        """Only the fields actually set, ready to merge into the knobs
        kwargs for Portfolio.configs()."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SegmentPolicyMap(BaseModel):
    """`default` applies to any segment id not in `segments`, and fills in
    any field a matched segment entry leaves unset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default: SegmentPolicyEntry = SegmentPolicyEntry()
    segments: dict[str, SegmentPolicyEntry] = {}

    def resolve(self, segment_id: str) -> dict[str, float | int | str]:
        """Knobs for one segment id: default entry's fields, overridden
        field-by-field by the matching segment entry (if any)."""
        knobs = self.default.as_knobs()
        knobs.update(self.segments.get(segment_id, SegmentPolicyEntry()).as_knobs())
        return knobs

    def merge(self, overrides: "SegmentPolicyMap") -> "SegmentPolicyMap":
        """Runtime overrides win field-by-field, per segment. A segment
        present only in `overrides` is added; one present in both keeps
        this map's fields except where `overrides` sets its own."""
        merged_default = self.default.model_copy(
            update=overrides.default.model_dump(exclude_none=True)
        )
        merged_segments = dict(self.segments)
        for seg_id, entry in overrides.segments.items():
            base = merged_segments.get(seg_id, SegmentPolicyEntry())
            merged_segments[seg_id] = base.model_copy(update=entry.model_dump(exclude_none=True))
        return SegmentPolicyMap(default=merged_default, segments=merged_segments)


def load_segment_policy_map(path: str | Path) -> SegmentPolicyMap:
    """Load a SegmentPolicyMap from YAML. Shape:

        default:
          factor: 1.65
          method: sqrt_horizon
          mode: base_stock
        segments:
          "abc=A|demand_pattern=lumpy":
            factor: 2.0
            method: k_rmse
          "group=kit":
            mode: rop
            factor: 0.8

    `method` must be one of sqrt_horizon, k_rmse, k_mae (Portfolio.configs'
    safety_stock_method); `mode` one of base_stock, rop.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    return SegmentPolicyMap.model_validate(data)
