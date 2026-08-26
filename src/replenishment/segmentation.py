"""Segmentation: assign every item a composite segment key BEFORE policy
selection, so which SafetyStockStrategy/knobs an item gets is a lookup
against its segment, not a per-item ad-hoc choice.

A segment key is built by running a handful of independent SegmentRules
over the same portfolio, each contributing one label (e.g. demand_pattern
-> "lumpy", abc -> "A"). Rules are pluggable: ABC/XYZ/demand-pattern are
data-driven, AttributeRule carries a business-assigned tag, and
ExplicitGroupRule pins a set of items that must share a policy for a
reason no data rule would infer ("these two SKUs ship in one kit").
Nothing in this module chooses policy knobs, see segment_policy.py for
the segment -> knobs mapping.

    rules = [DemandPatternRule(), ABCRule()]
    keys = segment_portfolio(rows, rules)
    keys["sku-1"].id  # "abc=A|demand_pattern=smooth"
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from replenishment.classification import classify_abc, classify_demand, classify_xyz
from replenishment.io_ import StandardSimulationRow

# Free-form extra data a rule may need beyond the rows themselves (e.g. a
# caller-supplied item-value table). Rules that don't need it ignore it.
SegmentContext = Mapping[str, Any]


def _group_by_unique_id(
    rows: Iterable[StandardSimulationRow],
) -> dict[str, list[StandardSimulationRow]]:
    grouped: dict[str, list[StandardSimulationRow]] = {}
    for row in rows:
        grouped.setdefault(row.unique_id, []).append(row)
    return grouped


class SegmentRule(Protocol):
    """One axis of segmentation. Handed the whole portfolio (not one item
    at a time) because rules like ABCRule need cross-item ranking to label
    even a single item."""

    name: str

    def assign_all(
        self,
        rows_by_item: Mapping[str, list[StandardSimulationRow]],
        context: SegmentContext,
    ) -> dict[str, str]:
        """unique_id -> this rule's label for that item. Must cover every
        key in rows_by_item."""
        ...


@dataclass(frozen=True)
class DemandPatternRule:
    """Labels each item by its Syntetos-Boylan demand class (smooth /
    erratic / intermittent / lumpy), via classify_demand on its demand
    history sorted by ds."""

    name: str = "demand_pattern"

    def assign_all(
        self,
        rows_by_item: Mapping[str, list[StandardSimulationRow]],
        context: SegmentContext,
    ) -> dict[str, str]:
        result = {}
        for uid, rows in rows_by_item.items():
            history = [r.demand for r in sorted(rows, key=lambda r: r.ds)]
            result[uid] = classify_demand(history)
        return result


@dataclass(frozen=True)
class ABCRule:
    """Labels each item by cumulative-value tier (A/B/C). value_fn defaults
    to total holding-cost-weighted demand (a stand-in for $ usage); pass
    your own for real revenue/margin data."""

    cutoffs: tuple[float, float] = (0.8, 0.95)
    value_fn: Callable[[list[StandardSimulationRow]], float] = field(
        default=lambda rows: sum(r.demand * r.holding_cost_per_unit for r in rows)
    )
    name: str = "abc"

    def assign_all(
        self,
        rows_by_item: Mapping[str, list[StandardSimulationRow]],
        context: SegmentContext,
    ) -> dict[str, str]:
        values = {uid: self.value_fn(rows) for uid, rows in rows_by_item.items()}
        return classify_abc(values, cutoffs=self.cutoffs)


@dataclass(frozen=True)
class XYZRule:
    """Labels each item by demand-variability tier (X/Y/Z), via the
    coefficient of variation of its per-period demand."""

    thresholds: tuple[float, float] = (0.5, 1.0)
    name: str = "xyz"

    def assign_all(
        self,
        rows_by_item: Mapping[str, list[StandardSimulationRow]],
        context: SegmentContext,
    ) -> dict[str, str]:
        result = {}
        for uid, rows in rows_by_item.items():
            demand = [r.demand for r in rows]
            mean = sum(demand) / len(demand) if demand else 0.0
            if len(demand) > 1 and mean > 0:
                cv = statistics.stdev(demand) / mean
            else:
                cv = 0.0
            result[uid] = classify_xyz(cv, thresholds=self.thresholds)
        return result


@dataclass(frozen=True)
class AttributeRule:
    """Labels each item from a caller-supplied business tag (e.g. supplier,
    channel, planner-assigned priority) that no data rule could infer.
    Items missing from `attributes` get `default_label`."""

    attributes: Mapping[str, str]
    default_label: str = "unassigned"
    name: str = "attribute"

    def assign_all(
        self,
        rows_by_item: Mapping[str, list[StandardSimulationRow]],
        context: SegmentContext,
    ) -> dict[str, str]:
        return {uid: self.attributes.get(uid, self.default_label) for uid in rows_by_item}


@dataclass(frozen=True)
class ExplicitGroupRule:
    """Pins a set of items to a shared label, for "must optimize together"
    constraints a data or business-tag rule wouldn't produce on its own
    (e.g. components that ship as one kit). Items not named in `groups`
    each keep their own identity (labeled by unique_id) so they're never
    accidentally merged with anything."""

    groups: Mapping[str, str]
    name: str = "group"

    def assign_all(
        self,
        rows_by_item: Mapping[str, list[StandardSimulationRow]],
        context: SegmentContext,
    ) -> dict[str, str]:
        return {uid: self.groups.get(uid, uid) for uid in rows_by_item}


@dataclass(frozen=True)
class SegmentKey:
    """Composite segment label: rule name -> that rule's label for one
    item. `.id` is the deterministic string used as a SegmentPolicyMap key
    (rule names sorted, so rule order never changes the id)."""

    labels: Mapping[str, str]

    @property
    def id(self) -> str:
        return "|".join(f"{name}={label}" for name, label in sorted(self.labels.items()))


def segment_portfolio(
    rows: Iterable[StandardSimulationRow],
    rules: Sequence[SegmentRule],
    context: SegmentContext | None = None,
) -> dict[str, SegmentKey]:
    """Run every rule over the portfolio and build one SegmentKey per item.

    Args:
        rows: the portfolio's StandardSimulationRows (all items, all periods).
        rules: applied in order given; SegmentKey.id sorts by name regardless.
        context: optional extra data passed through to every rule.

    Returns:
        unique_id -> SegmentKey.
    """
    rows_by_item = _group_by_unique_id(rows)
    if not rows_by_item:
        return {}
    ctx = context or {}
    per_rule: dict[str, dict[str, str]] = {}
    for rule in rules:
        labels = rule.assign_all(rows_by_item, ctx)
        missing = rows_by_item.keys() - labels.keys()
        if missing:
            raise ValueError(f"Rule {rule.name!r} did not label items: {sorted(missing)}.")
        per_rule[rule.name] = labels

    return {
        uid: SegmentKey(labels={rule_name: labels[uid] for rule_name, labels in per_rule.items()})
        for uid in rows_by_item
    }
