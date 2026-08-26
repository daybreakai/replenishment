# Duvo Pattern Adoption — Typed Report & Degradation-Ladder Design Spec

**Date:** 2026-08-26
**Status:** Proposed
**Origin:** Competitive-capture review of Duvo.ai's "Inventory Optimization" Claude Skill (captured 2026-08-17 to `~/Downloads/duvo-inventory-optimization-skill/`), compared against `daybreakai/replenishment` at the user's request to steal what's worth stealing, including typed pydantic outputs.
**Target branch:** Local `main` (tip `7ee509e` at authoring time), not `sbd-poc3-local-fixtures`. This spec was drafted by reading `sbd-poc3-local-fixtures`'s working tree, which turned out to be several commits behind local `main` (main added `portfolio.py`, `classification.py`, `strategies/distributional_safety_stock.py`, `FixedErrorSafetyStock`, and `moq`/`base_stock_flat` policy support that the initial draft never saw). §2 and §4 below are corrected for main's actual current state — this is a pre-flight-scan ruling, not a rewrite of the plan's intent.

## 1. What Duvo actually is (and isn't)

Duvo's dir is a **prompt-time Claude Skill**: `SKILL.md` (a 6-stage methodology the LLM executes ad hoc) + 6 reference docs (formulas/thresholds) + 2 helper scripts (`classify-inventory.py`, `calculate-stock-health.py` — argparse, `dataclasses`, `csv.DictReader`, `print()`-formatted markdown tables) + 3 SQL extraction templates. No pydantic anywhere. No persisted domain model, no typed output contract, no tests, no scoring loop — the `README.md` already captured in that dir (prior internal competitive-capture note) says as much: "the entire capability is a prompt-time skill... There is no persisted domain model, no decision record, no scoring loop."

`daybreakai/replenishment` is a different shape of product: a pip-installed, pytest-tested, notebook-consumed **compute library** (`TimeSeries`, `SafetyStockStrategy`, `OrderTrigger`, `ReplenishmentPolicy`, `simulate_replenishment`, `calibration.optimize`). Mirroring Duvo's directory layout 1:1 (`SKILL.md` / `references/` / `scripts/`) would be cargo-culting the wrong shape onto this repo — there is no agent runtime here to execute a `SKILL.md` against. That packaging is worth revisiting only if/when this library grows an agent-facing surface (e.g. feeding a demand-planner-agent chat or report task); premature now.

## 2. What's worth stealing

Three patterns, all present in Duvo's design and absent (or only implicit) in ours today:

**a. Degradation ladder.** `calculate-stock-health.py::calculate_safety_stock()` cascades: full formula (σ_d and σ_LT both known) → σ_d-only formula → flat 25% buffer — never blocking on missing data. We already have the *equivalent strategies* as separate classes (`SqrtHorizonSafetyStock`, `KRmseSafetyStock`, `KMaeSafetyStock`, `FillRateSafetyStock`, `MultiplierSafetyStockStrategy`, `NullSafetyStockStrategy`, plus, as of main's current tip, `FixedErrorSafetyStock`, `KingsFormulaSafetyStock`, `CompoundPoissonSafetyStock`), but nothing in this repo *picks among them* based on what data is actually available. That selection function doesn't exist yet.

  Main already has one adjacent selector, `classification.py::classify_demand()` — a Syntetos-Boylan router that picks among strategies by **demand shape** (smooth/erratic/intermittent/lumpy: ADI/CV² thresholds). That is an orthogonal axis to Duvo's ladder, which degrades by **data availability** (do we have σ_d? σ_LT? actuals at all?). This plan's resolver stays scoped to the data-availability axis and does not fold in `classify_demand()` — merging two independent selection axes into one function is its own design problem, not implied by "steal the degradation-ladder pattern." A caller free to combine both (classify demand shape, then resolve data availability within that shape's preferred strategy) already can, since both return plain values/instances with no shared state.

**b. Typed, validated output instead of `print()`-to-stdout markdown.** Duvo's scripts hand-format f-strings straight to stdout; nothing stops a `NaN`, a wrong-shaped record, or a silently-`None` field from reaching the table. `pydantic` models give a validated, JSON-serializable output contract any downstream consumer (notebook, future agent, report renderer) can rely on — this is the direct answer to "typed pydantic outputs where possible."

**c. Stages-applied / stages-skipped transparency.** Duvo's final report always states which of its 6 stages ran vs. were skipped and why, so the reader calibrates confidence in what they're looking at. Our report should do the same for whichever health dimensions it can and can't compute from what's wired in today (see §5, non-goals — several dimensions genuinely can't run yet, and the report should say so instead of omitting silently).

## 3. What's explicitly not adopted

- **Retail category DOS thresholds** (fresh/frozen/ambient/non-food bands) — hardcoded and grocery-specific. This repo's thresholds are caller-supplied, never baked in.
- **The `SKILL.md`/`references`/`scripts` packaging itself** — see §1.
- **ABC/XYZ segmentation** (revenue-based Pareto classification) — still absent from this codebase (`classify_demand()` classifies demand *pattern*, not revenue tier — a different concept). Out of scope here; would be a new module, not a report-layer change.

## 4. Design

### `strategies/resolver.py`

```python
def resolve_safety_stock_strategy(
    *,
    has_actuals: bool,
    periods_observed: int,
    target_fill_rate: float | None = None,
    factor: float | None = None,
    multiplier: float | None = None,
) -> ResolvedSafetyStock: ...
```

Ladder (mirrors Duvo's cascade, expressed in this repo's own strategy vocabulary — highest-fidelity method first, each rung a graceful fallback from the one above):

1. `has_actuals` and `periods_observed > 0` and `target_fill_rate` given → `FillRateSafetyStock(target_fill_rate)`. Exact service-level semantics; the highest-fidelity method available.
2. `has_actuals` and `periods_observed > 0` and `factor` given → `SqrtHorizonSafetyStock(factor)`. Scales with lead time + horizon; general-purpose when no fill-rate target is set.
3. `has_actuals` and `periods_observed > 0`, no `factor` given → `KRmseSafetyStock(factor=1.0)`, `degraded=True`. Flat fallback — mirrors Duvo's "flat 25% buffer when σ_d is unknown" rung, just expressed with this repo's own strategy instead of a hardcoded percentage.
4. no actuals, `multiplier` given → `MultiplierSafetyStockStrategy(multiplier)`, `degraded=True`.
5. none of the above → `NullSafetyStockStrategy()`, `degraded=True`, reason logged. Never raises — mirrors Duvo's "never block" principle.

`ResolvedSafetyStock` (pydantic model): `strategy` (the constructed strategy instance — `arbitrary_types_allowed`), `method: str` (class name), `degraded: bool`, `reason: str | None`.

### `report.py`

Pydantic models `PolicyHealth`, `PortfolioSummary`, `ReplenishmentReport`, plus:

```python
def build_report(entries: list[PolicyRun]) -> ReplenishmentReport: ...
```

where `PolicyRun` pairs a `label: str` with a `SimulationResult` (from `simulation.py`) and a `ResolvedSafetyStock` (from `resolver.py`). `ReplenishmentReport.to_markdown()` renders the same shape of table + summary Duvo's scripts print to stdout, but built from a validated model instead of raw string formatting — same reader-facing output, typed contract underneath.

**Reuse, not reinvention, of the existing portfolio layer.** Main already has `portfolio.py::Portfolio`/`PortfolioResult` — the real front door for multi-item simulation, with `summary_frame()` (a pandas DataFrame: fill_rate/costs/avg_on_hand per item) and `portfolio_metrics()` (weighted_fill, worst_month_fill, cost totals). `report.py` does not re-implement that aggregation. Instead it adds one adapter:

```python
def policy_runs_from_portfolio(
    portfolio_result: PortfolioResult,
    resolved_strategies: Mapping[str, ResolvedSafetyStock],
) -> list[PolicyRun]: ...
```

which zips `PortfolioResult.results` (`dict[str, SimulationResult]`, keyed by `unique_id`) against a caller-supplied `{unique_id: ResolvedSafetyStock}` mapping into `PolicyRun`s ready for `build_report`. `Portfolio`/`PortfolioResult` stay the numeric aggregation surface (DataFrame, dict scorecard); `report.py` is the typed, validated, stage-transparent *narrative* surface on top — genuinely additive, not a parallel reimplementation. Wiring `resolved_strategies` itself (e.g. from `Portfolio`'s existing string-keyed `method=` builder param in `io_.py`) is left to the caller — reconciling the two safety-stock-selection mechanisms (`io_.py`'s string dispatch vs. this plan's `resolve_safety_stock_strategy`) is a separate integration effort, noted under Non-goals.

## 5. Non-goals

- **No overstock / dead-stock / ABC-XYZ stages** in `build_report` — those need inputs (unit cost, category, revenue) not modeled anywhere in `policy.py`/`io_.py`/`portfolio.py` today. `build_report` lists them explicitly under `PortfolioSummary.stages_skipped` with the reason, applying the graceful-degradation pattern (§2a/c) to its own report rather than faking a stage it can't actually run.
- **No reconciliation of `resolve_safety_stock_strategy` with `io_.py`'s existing string-keyed `safety_stock_method` dispatch** (the mechanism `Portfolio.configs(method=...)` already uses to pick a strategy when building `ArticleSimulationConfig`s). Both are legitimate selectors on different inputs (data-availability vs. a caller-chosen method name); merging them is a follow-up, not this plan.
