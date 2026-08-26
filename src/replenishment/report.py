"""Typed, validated inventory-health report. Replaces the print()-to-stdout
markdown pattern in Duvo.ai's calculate-stock-health.py / classify-inventory.py
with a pydantic contract, and carries forward their stages-applied /
stages-skipped transparency principle: dimensions this repo can't compute
yet (overstock, dead-stock, ABC/XYZ -- see spec Sec. 5) are listed with a
reason instead of omitted silently.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from replenishment.portfolio import PortfolioResult
from replenishment.simulation import SimulationResult
from replenishment.strategies.resolver import ResolvedSafetyStock, resolve_safety_stock_strategy

HealthStatus = Literal["Healthy", "Understock Risk", "No Data"]

UNDERSTOCK_FILL_RATE_THRESHOLD = 0.90

SKIPPED_STAGES = [
    "overstock (no unit_cost/category input wired into ReplenishmentPolicy yet)",
    "dead-stock (no last-movement-date input wired in yet)",
    "ABC/XYZ segmentation (portfolio-level; out of scope for a single-policy run)",
]


@dataclass(frozen=True)
class PolicyRun:
    """One item's simulation result paired with the safety-stock strategy
    resolve_safety_stock_strategy picked for it, ready to feed build_report()."""

    label: str
    result: SimulationResult
    resolved_safety_stock: ResolvedSafetyStock


class PolicyHealth(BaseModel):
    """One item's row in a ReplenishmentReport."""

    model_config = ConfigDict(frozen=True)

    label: str
    fill_rate: float
    avg_on_hand: float
    total_cost: float
    safety_stock_method: str
    degraded: bool
    degradation_reason: str | None
    health_status: HealthStatus


class PortfolioSummary(BaseModel):
    """Aggregate counts and stage transparency for a ReplenishmentReport."""

    model_config = ConfigDict(frozen=True)

    total_policies: int
    status_counts: dict[HealthStatus, int]
    stages_applied: list[str]
    stages_skipped: list[str]


class ReplenishmentReport(BaseModel):
    """A validated, typed replenishment health report: per-item records plus
    a portfolio-level summary. See build_report() to construct one."""

    model_config = ConfigDict(frozen=True)

    records: list[PolicyHealth]
    summary: PortfolioSummary

    def to_markdown(self) -> str:
        """Render the same shape of report Duvo's scripts print()ed to
        stdout, but from this validated model instead of raw string
        formatting."""
        lines = ["## Replenishment Health Report", ""]
        lines.append(f"**Stages applied:** {', '.join(self.summary.stages_applied)}")
        lines.append(f"**Stages skipped:** {'; '.join(self.summary.stages_skipped)}")
        lines.append("")
        lines.append("| Label | Fill Rate | Avg On-Hand | Total Cost | SS Method | Degraded | Status |")
        lines.append("|-------|-----------|-------------|------------|-----------|----------|--------|")
        for r in self.records:
            label = r.label.replace("|", "\\|")
            lines.append(
                f"| {label} | {r.fill_rate:.1%} | {r.avg_on_hand:.1f} | "
                f"${r.total_cost:,.2f} | {r.safety_stock_method} | {r.degraded} | {r.health_status} |"
            )
        lines.append("")
        lines.append("## Summary by Health Status")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        for status, count in self.summary.status_counts.items():
            lines.append(f"| {status} | {count} |")
        lines.append("")
        lines.append(f"**Total policies analyzed:** {self.summary.total_policies}")
        return "\n".join(lines)


def _health_status(fill_rate: float, total_demand: int, understock_fill_rate_threshold: float) -> HealthStatus:
    if total_demand == 0:
        return "No Data"
    if not math.isfinite(fill_rate):
        return "No Data"  # a non-finite fill rate is a broken metric, never "Healthy"
    if fill_rate < understock_fill_rate_threshold:
        return "Understock Risk"
    return "Healthy"


def build_report(
    entries: list[PolicyRun],
    *,
    understock_fill_rate_threshold: float = UNDERSTOCK_FILL_RATE_THRESHOLD,
) -> ReplenishmentReport:
    """Build a ReplenishmentReport from per-item PolicyRuns. Always lists
    SKIPPED_STAGES under summary.stages_skipped (overstock/dead-stock/ABC-XYZ
    -- see module docstring) rather than omitting them silently."""
    records: list[PolicyHealth] = []
    for entry in entries:
        summary = entry.result.summary
        status = _health_status(summary.fill_rate, summary.total_demand, understock_fill_rate_threshold)
        records.append(PolicyHealth(
            label=entry.label,
            fill_rate=summary.fill_rate,
            avg_on_hand=summary.average_on_hand,
            total_cost=summary.total_cost,
            safety_stock_method=entry.resolved_safety_stock.method,
            degraded=entry.resolved_safety_stock.degraded,
            degradation_reason=entry.resolved_safety_stock.reason,
            health_status=status,
        ))

    status_counts: dict[HealthStatus, int] = {}
    for r in records:
        status_counts[r.health_status] = status_counts.get(r.health_status, 0) + 1

    summary = PortfolioSummary(
        total_policies=len(records),
        status_counts=status_counts,
        stages_applied=["health"],
        stages_skipped=list(SKIPPED_STAGES),
    )
    return ReplenishmentReport(records=records, summary=summary)


def policy_runs_from_portfolio(
    portfolio_result: PortfolioResult,
    resolved_strategies: Mapping[str, ResolvedSafetyStock],
) -> list[PolicyRun]:
    """Adapter from the existing Portfolio/PortfolioResult front door
    (portfolio.py) into this module's PolicyRun/build_report. Does not
    reimplement PortfolioResult's own aggregation (summary_frame(),
    portfolio_metrics()) -- this only re-keys its per-item SimulationResults
    against a caller-supplied {unique_id: ResolvedSafetyStock} mapping so
    they can feed build_report().

    An id missing from resolved_strategies does not raise -- it degrades to
    a NullSafetyStockStrategy (via resolve_safety_stock_strategy) with a
    reason naming the missing id, matching this module's never-omit-silently
    principle rather than crashing on a caller's incomplete mapping.
    """
    runs = []
    for unique_id, result in portfolio_result.results.items():
        resolved = resolved_strategies.get(unique_id)
        if resolved is None:
            resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
            resolved = resolved.model_copy(update={
                "reason": f"No resolved safety-stock strategy supplied for {unique_id!r}; defaulted to Null.",
            })
        runs.append(PolicyRun(label=unique_id, result=result, resolved_safety_stock=resolved))
    return runs
