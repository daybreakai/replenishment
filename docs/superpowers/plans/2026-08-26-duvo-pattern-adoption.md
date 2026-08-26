# Duvo Pattern Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a degradation-ladder resolver that picks a `SafetyStockStrategy` from data availability (mirroring Duvo.ai's inventory-skill cascade), and a typed `pydantic` report layer that replaces ad hoc print-to-stdout output with a validated, JSON-serializable contract — without adopting Duvo's `SKILL.md`/prompt-execution packaging, which is the wrong shape for this repo.

**Architecture:** Two new, independent modules on top of the existing strategy/simulation layers. `strategies/resolver.py` depends only on `strategies/safety_stock.py` and `strategies/multiplier.py` (already-existing strategy classes — no new strategies). `report.py` depends on `simulation.py`'s `SimulationResult`/`SimulationSummary` and `resolver.py`'s `ResolvedSafetyStock`. Neither module changes `policy.py`, `simulation.py`, or any existing public API — both are additive.

**Tech Stack:** Python 3.11+, `pydantic>=2` (new dependency), `pytest` (existing).

**Spec:** `docs/superpowers/specs/2026-08-26-duvo-pattern-adoption-design.md`

**Branch:** Local `main` (tip `7ee509e` at authoring time) — NOT `sbd-poc3-local-fixtures`. Pre-flight discovery: local `main` is ahead of that branch (which is an ancestor of main, merge-base = `c5434c6`) and already carries `portfolio.py`, `classification.py`, `strategies/distributional_safety_stock.py`, `FixedErrorSafetyStock`, and `moq`/`base_stock_flat` policy support. Both tasks below are written against main's actual current API; see the spec's "Target branch" note and Sec. 4's "Reuse, not reinvention" note for what changed from the first draft.

## Global Constraints

- No retail-specific hardcoded thresholds (DOS bands, category tables) — this repo's thresholds are always caller-supplied.
- No ABC/XYZ segmentation, no overstock/dead-stock stages — out of scope (single-policy library, no portfolio-level cost/category inputs).
- `resolve_safety_stock_strategy` never raises for missing data — it always returns a `ResolvedSafetyStock`, falling back to `NullSafetyStockStrategy` with `degraded=True` as the last rung.
- `build_report` must list any stage it cannot compute (overstock, dead-stock, ABC/XYZ) under `PortfolioSummary.stages_skipped` with a reason — never omit silently.
- New code follows existing repo conventions: `from __future__ import annotations`, frozen dataclasses stay dataclasses; only the new report/resolver *output* models are pydantic — do not convert existing dataclasses (`SimulationResult`, `TimeSeries`, etc.) to pydantic.

---

## File Structure

```
src/replenishment/
    strategies/
        resolver.py      # NEW: resolve_safety_stock_strategy(...) -> ResolvedSafetyStock
    report.py             # NEW: PolicyHealth, PortfolioSummary, ReplenishmentReport,
                           #      PolicyRun, build_report(...), policy_runs_from_portfolio(...)

tests/
    strategies/
        test_resolver.py  # NEW
    test_report.py        # NEW

pyproject.toml             # MODIFY: add pydantic>=2 to [project].dependencies
```

---

### Task 1: Degradation-ladder resolver

**Files:**
- Create: `src/replenishment/strategies/resolver.py`
- Modify: `pyproject.toml` (add `pydantic>=2` to `dependencies`)
- Test: `tests/strategies/test_resolver.py`

**Interfaces:**
- Consumes: `SqrtHorizonSafetyStock`, `KRmseSafetyStock`, `FillRateSafetyStock` from `replenishment.strategies.safety_stock`; `MultiplierSafetyStockStrategy`, `NullSafetyStockStrategy` from `replenishment.strategies.multiplier`.
- Produces: `ResolvedSafetyStock` (pydantic `BaseModel` with fields `strategy: object`, `method: str`, `degraded: bool`, `reason: str | None`) and `resolve_safety_stock_strategy(*, has_actuals: bool, periods_observed: int, target_fill_rate: float | None = None, factor: float | None = None, multiplier: float | None = None) -> ResolvedSafetyStock`, both importable from `replenishment.strategies.resolver`. `report.py` (Task 2) consumes both.

- [ ] **Step 1: Add the `pydantic` dependency**

Edit `pyproject.toml`'s `[project]` `dependencies` list to add `"pydantic>=2"`:

```toml
dependencies = [
    "pandas>=2.0",
    "matplotlib>=3.7",
    "pyarrow>=25.0.1",
    "pydantic>=2",
]
```

Run: `uv sync`
Expected: resolves and installs `pydantic` into `.venv` with no conflicts.

- [ ] **Step 2: Write the failing tests**

Create `tests/strategies/test_resolver.py`:

```python
from replenishment.strategies.resolver import resolve_safety_stock_strategy
from replenishment.strategies.safety_stock import (
    FillRateSafetyStock, SqrtHorizonSafetyStock, KRmseSafetyStock,
)
from replenishment.strategies.multiplier import (
    MultiplierSafetyStockStrategy, NullSafetyStockStrategy,
)


def test_prefers_fill_rate_when_target_and_actuals_available():
    resolved = resolve_safety_stock_strategy(
        has_actuals=True, periods_observed=10, target_fill_rate=0.95, factor=1.65,
    )
    assert isinstance(resolved.strategy, FillRateSafetyStock)
    assert resolved.method == "FillRateSafetyStock"
    assert resolved.degraded is False


def test_falls_back_to_sqrt_horizon_when_no_fill_rate_target():
    resolved = resolve_safety_stock_strategy(
        has_actuals=True, periods_observed=10, target_fill_rate=None, factor=1.65,
    )
    assert isinstance(resolved.strategy, SqrtHorizonSafetyStock)
    assert resolved.degraded is False


def test_falls_back_to_flat_k_rmse_when_no_factor_given():
    resolved = resolve_safety_stock_strategy(
        has_actuals=True, periods_observed=10, target_fill_rate=None, factor=None,
    )
    assert isinstance(resolved.strategy, KRmseSafetyStock)
    assert resolved.strategy.factor == 1.0
    assert resolved.degraded is True
    assert "flat" in resolved.reason.lower()


def test_falls_back_to_multiplier_when_no_actuals():
    resolved = resolve_safety_stock_strategy(
        has_actuals=False, periods_observed=0, multiplier=1.2,
    )
    assert isinstance(resolved.strategy, MultiplierSafetyStockStrategy)
    assert resolved.degraded is True


def test_falls_back_to_null_when_nothing_available():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    assert isinstance(resolved.strategy, NullSafetyStockStrategy)
    assert resolved.degraded is True
    assert resolved.reason is not None


def test_no_actuals_but_periods_observed_positive_is_not_enough():
    # has_actuals=False must win even if periods_observed is nonzero (defensive:
    # caller passed a stale count without actuals wired in).
    resolved = resolve_safety_stock_strategy(
        has_actuals=False, periods_observed=10, factor=1.65, target_fill_rate=0.95,
    )
    assert isinstance(resolved.strategy, NullSafetyStockStrategy)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/strategies/test_resolver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replenishment.strategies.resolver'`

- [ ] **Step 4: Implement the resolver**

Create `src/replenishment/strategies/resolver.py`:

```python
"""Degradation-ladder selection of a SafetyStockStrategy from data
availability. Mirrors the cascade in Duvo.ai's inventory-optimization
skill (calculate-stock-health.py::calculate_safety_stock): full formula
-> simpler formula -> flat buffer -> never block -- expressed here with
this repo's own SafetyStockStrategy classes instead of a hardcoded
percentage.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from replenishment.strategies.multiplier import MultiplierSafetyStockStrategy, NullSafetyStockStrategy
from replenishment.strategies.safety_stock import FillRateSafetyStock, KRmseSafetyStock, SqrtHorizonSafetyStock


class ResolvedSafetyStock(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: object
    method: str
    degraded: bool
    reason: str | None = None


def resolve_safety_stock_strategy(
    *,
    has_actuals: bool,
    periods_observed: int,
    target_fill_rate: float | None = None,
    factor: float | None = None,
    multiplier: float | None = None,
) -> ResolvedSafetyStock:
    if has_actuals and periods_observed > 0:
        if target_fill_rate is not None:
            strategy = FillRateSafetyStock(target_fill_rate=target_fill_rate)
            return ResolvedSafetyStock(strategy=strategy, method=type(strategy).__name__, degraded=False)
        if factor is not None:
            strategy = SqrtHorizonSafetyStock(factor=factor)
            return ResolvedSafetyStock(strategy=strategy, method=type(strategy).__name__, degraded=False)
        strategy = KRmseSafetyStock(factor=1.0)
        return ResolvedSafetyStock(
            strategy=strategy, method=type(strategy).__name__, degraded=True,
            reason="No target_fill_rate or factor given; using a flat K-RMSE buffer (factor=1.0) "
                   "as the last resolvable rung before Null.",
        )

    if multiplier is not None:
        strategy = MultiplierSafetyStockStrategy(multiplier=multiplier)
        return ResolvedSafetyStock(
            strategy=strategy, method=type(strategy).__name__, degraded=True,
            reason="No actuals available to compute forecast error; using a flat multiplier instead.",
        )

    strategy = NullSafetyStockStrategy()
    return ResolvedSafetyStock(
        strategy=strategy, method=type(strategy).__name__, degraded=True,
        reason="No actuals and no multiplier given; safety stock defaulted to 0 -- treat with caution.",
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/strategies/test_resolver.py -v`
Expected: PASS (6/6)

- [ ] **Step 6: Commit**

`uv.lock` is gitignored on this branch (see `.gitignore`) — do not add it.

```bash
git add pyproject.toml src/replenishment/strategies/resolver.py tests/strategies/test_resolver.py
git commit -m "feat: add degradation-ladder safety-stock strategy resolver"
```

---

### Task 2: Typed pydantic report layer

**Files:**
- Create: `src/replenishment/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `SimulationResult`, `SimulationSummary` from `replenishment.simulation`; `ResolvedSafetyStock` from `replenishment.strategies.resolver` (Task 1); `PortfolioResult` from `replenishment.portfolio` (already exists on `main` — do not modify `portfolio.py`).
- Produces: `PolicyRun` (plain dataclass: `label: str`, `result: SimulationResult`, `resolved_safety_stock: ResolvedSafetyStock`), `PolicyHealth`, `PortfolioSummary`, `ReplenishmentReport` (pydantic `BaseModel`s), `build_report(entries: list[PolicyRun]) -> ReplenishmentReport`, and `policy_runs_from_portfolio(portfolio_result: PortfolioResult, resolved_strategies: Mapping[str, ResolvedSafetyStock]) -> list[PolicyRun]`, all importable from `replenishment.report`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
from replenishment.report import PolicyRun, build_report
from replenishment.strategies.resolver import resolve_safety_stock_strategy
from replenishment.simulation import InventorySnapshot, SimulationResult, SimulationSummary


def _result(fill_rate: float, total_demand: int = 100) -> SimulationResult:
    summary = SimulationSummary(
        total_demand=total_demand, total_fulfilled=int(total_demand * fill_rate),
        total_backorders=total_demand - int(total_demand * fill_rate), fill_rate=fill_rate,
        average_on_hand=50.0, holding_cost=50.0, stockout_cost=0.0,
        ordering_cost=10.0, total_cost=60.0,
    )
    return SimulationResult(snapshots=[], summary=summary, ending_pipeline=[])


def test_build_report_marks_healthy_policy():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku-1", result=_result(fill_rate=0.98), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert len(report.records) == 1
    assert report.records[0].health_status == "Healthy"
    assert report.records[0].safety_stock_method == "SqrtHorizonSafetyStock"
    assert report.records[0].degraded is False


def test_build_report_flags_understock_risk_below_90pct_fill_rate():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    entry = PolicyRun(label="sku-2", result=_result(fill_rate=0.80), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert report.records[0].health_status == "Understock Risk"
    assert report.records[0].degraded is True


def test_build_report_no_data_status_for_zero_demand():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    entry = PolicyRun(label="sku-3", result=_result(fill_rate=1.0, total_demand=0), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert report.records[0].health_status == "No Data"


def test_build_report_summary_counts_and_skipped_stages():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entries = [
        PolicyRun(label="sku-1", result=_result(fill_rate=0.98), resolved_safety_stock=resolved),
        PolicyRun(label="sku-2", result=_result(fill_rate=0.98), resolved_safety_stock=resolved),
    ]
    report = build_report(entries)
    assert report.summary.total_policies == 2
    assert report.summary.status_counts["Healthy"] == 2
    assert "overstock" in " ".join(report.summary.stages_skipped).lower()


def test_to_markdown_renders_a_table_with_all_labels():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku-1", result=_result(fill_rate=0.98), resolved_safety_stock=resolved)
    report = build_report([entry])
    markdown = report.to_markdown()
    assert "sku-1" in markdown
    assert "Healthy" in markdown


def test_policy_runs_from_portfolio_zips_results_with_resolved_strategies():
    from replenishment.portfolio import PortfolioResult
    from replenishment.report import policy_runs_from_portfolio

    resolved_a = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    resolved_b = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    portfolio_result = PortfolioResult(results={
        "sku-a": _result(fill_rate=0.98),
        "sku-b": _result(fill_rate=0.80),
    })
    runs = policy_runs_from_portfolio(
        portfolio_result, {"sku-a": resolved_a, "sku-b": resolved_b},
    )
    by_label = {r.label: r for r in runs}
    assert by_label["sku-a"].resolved_safety_stock is resolved_a
    assert by_label["sku-b"].resolved_safety_stock is resolved_b
    report = build_report(runs)
    assert {r.label for r in report.records} == {"sku-a", "sku-b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replenishment.report'`

- [ ] **Step 3: Implement the report layer**

Create `src/replenishment/report.py`:

```python
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


def _health_status(fill_rate: float, total_demand: int) -> HealthStatus:
    if total_demand == 0:
        return "No Data"
    if fill_rate < UNDERSTOCK_FILL_RATE_THRESHOLD:
        return "Understock Risk"
    return "Healthy"


def build_report(entries: list[PolicyRun]) -> ReplenishmentReport:
    records: list[PolicyHealth] = []
    for entry in entries:
        summary = entry.result.summary
        status = _health_status(summary.fill_rate, summary.total_demand)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: PASS (6/6)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all existing tests plus the 6 resolver tests and 6 report tests pass; no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/replenishment/report.py tests/test_report.py
git commit -m "feat: add typed pydantic report layer with stage transparency"
```

---

## Explicitly deferred (not in this plan)

- Wrapping `report.py`'s output in a Duvo-style `SKILL.md` for agent consumption — only worth doing once this library has an agent-facing surface to hang it on (see spec §1). Revisit then, don't build now.
- Overstock / dead-stock / ABC-XYZ stages — need unit-cost, category, and revenue inputs not modeled anywhere in `policy.py`/`io_.py` today. Adding them is a separate, larger plan (new fields on `ReplenishmentPolicy` or a new portfolio-level type), not a report-layer change.
- README.md update documenting `report.py`/`resolver.py` usage — small follow-up once the API is stable post-review, not bundled here to keep this plan's diff reviewable in two sittings.
