"""Typed, validated inventory-health report. Replaces the print()-to-stdout
markdown pattern in Duvo.ai's calculate-stock-health.py / classify-inventory.py
with a pydantic contract, and carries forward their stages-applied /
stages-skipped transparency principle: dimensions this repo can't compute
yet (overstock, dead-stock, ABC/XYZ -- see spec Sec. 5) are listed with a
reason instead of omitted silently.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from replenishment.portfolio import PortfolioResult
from replenishment.simulation import SimulationResult
from replenishment.strategies.resolver import ResolvedSafetyStock

HealthStatus = Literal["Healthy", "Understock Risk", "No Data"]

UNDERSTOCK_FILL_RATE_THRESHOLD = 0.90

SKIPPED_STAGES = [
    "overstock (no unit_cost/category input wired into ReplenishmentPolicy yet)",
    "dead-stock (no last-movement-date input wired in yet)",
    "ABC/XYZ segmentation (portfolio-level; out of scope for a single-policy run)",
]


@dataclass(frozen=True)
class PolicyRun:
    label: str
    result: SimulationResult
    resolved_safety_stock: ResolvedSafetyStock


class PolicyHealth(BaseModel):
    label: str
    fill_rate: float
    avg_on_hand: float
    total_cost: float
    safety_stock_method: str
    degraded: bool
    degradation_reason: str | None
    health_status: HealthStatus


class PortfolioSummary(BaseModel):
    total_policies: int
    status_counts: dict[str, int]
    stages_applied: list[str]
    stages_skipped: list[str]


class ReplenishmentReport(BaseModel):
    records: list[PolicyHealth]
    summary: PortfolioSummary

    def to_markdown(self) -> str:
        lines = ["## Replenishment Health Report", ""]
        lines.append(f"**Stages applied:** {', '.join(self.summary.stages_applied)}")
        lines.append(f"**Stages skipped:** {'; '.join(self.summary.stages_skipped)}")
        lines.append("")
        lines.append("| Label | Fill Rate | Avg On-Hand | Total Cost | SS Method | Degraded | Status |")
        lines.append("|-------|-----------|-------------|------------|-----------|----------|--------|")
        for r in self.records:
            lines.append(
                f"| {r.label} | {r.fill_rate:.1%} | {r.avg_on_hand:.1f} | "
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
    if fill_rate < understock_fill_rate_threshold:
        return "Understock Risk"
    return "Healthy"


def build_report(
    entries: list[PolicyRun],
    *,
    understock_fill_rate_threshold: float = UNDERSTOCK_FILL_RATE_THRESHOLD,
) -> ReplenishmentReport:
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

    status_counts: dict[str, int] = {}
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
    """
    return [
        PolicyRun(label=unique_id, result=result, resolved_safety_stock=resolved_strategies[unique_id])
        for unique_id, result in portfolio_result.results.items()
    ]
