---
name: propose-strategy-space
description: Turn a natural-language inventory hypothesis into a strategy space -- a JSON list of candidate {strategy, trigger, param_grid} entries, grounded in the real strategy registry and each item's actual demand classification -- then log it as an auditable per-customer YAML record. Usage: /propose-strategy-space --hypothesis "..." --data path.csv --customer acme
allowed-tools: Bash, Read, Write, Glob
---

# Propose a strategy space from a hypothesis

Turns "here's what I believe about this inventory problem" into a concrete,
checkable set of strategy/trigger/hyperparameter candidates for the
`optimize-replenishment-cost` skill to grid-search -- instead of guessing a
single config by eye. Classification is deterministic (a script); *which*
strategies the hypothesis implies is a reasoning step you do yourself,
informed by the demand-shape table and the registry below.

## Constants

- Classification script: `scripts/classify_items.py`
- Validation script: `scripts/validate_strategy_space.py`
- Audit-log script: `scripts/log_decision.py`
- Demand-shape lookup table: `../run-replenishment-backtest/references/strategy-selection.md`
- Strategy/trigger registry: `replenishment.strategies.describe(name)`,
  `list_safety_stock_strategies()`, `list_order_triggers()`
- Eval fixtures: `evals/hypotheses.jsonl`, graded by `evals/run_eval.py`
- Audit log: `experiments/strategy_space_decisions/<customer>/<timestamp>-<slug>.yaml`
  (repo root, one file per run -- many customers, many runs, never one growing file)

## Execution steps

1. **Classify the data**, not just the hypothesis:
   ```
   python .claude/skills/propose-strategy-space/scripts/classify_items.py --data path.csv
   ```
   Returns per-`unique_id` `{demand_class, n_periods, has_actuals, mean_demand,
   zero_share}`. If the source columns aren't named `actuals`/`initial_on_hand`/
   `lead_time`, pass `--actuals-field`/`--initial-on-hand-field`/
   `--lead-time-field` -- the exact same flags `run-replenishment-backtest`
   takes, resolved through the same `replenishment.io_.load_standard_simulation_rows`
   helper, so both skills read a differently-named column identically. If the
   hypothesis claims a demand shape ("it's lumpy") that
   the data doesn't actually classify as, say so explicitly -- don't silently
   defer to either source. This classifies fresh from `--data` every run; if
   `segment-inventory-items` has already run for this customer, its
   `adi_cv2_class` in `t_segmentation` is the same `classify_demand()` call
   against that customer's live Databricks source rather than this step's
   local file -- treat this step's output as authoritative when working from
   a local CSV/parquet (as this skill does), and reach for `t_segmentation`
   only if you're deliberately comparing against a segmentation run that
   used different/fresher source data.
2. **Map demand class(es) present -> candidate strategy families** using
   `../run-replenishment-backtest/references/strategy-selection.md` (the
   Syntetos-Boylan table). Then let the hypothesis's *stated goal* (minimize
   stockouts vs. cut holding cost vs. hit a specific service level) bias which
   params you sweep within each family -- e.g. "hit 95% service level" points
   at `FillRateSafetyStock(target_fill_rate=...)` bracketing 0.95, not a flat
   multiplier; "minimize holding cost" points at the low end of a factor
   range, not the high end.
3. **Write the strategy space as JSON**, one object with `hypothesis`,
   `data_summary`, `selection_rationale`, and `candidates`:
   ```json
   {
     "hypothesis": "<echoed back verbatim>",
     "data_summary": {"demand_classes_present": ["lumpy"], "n_items": 5},
     "selection_rationale": "5 items, all lumpy; hypothesis targets stockout reduction, so only the two distributional strategies were grid-searched -- the smooth-demand strategies (KRmse/SqrtHorizon/FillRate) don't fit this data and were excluded.",
     "candidates": [
       {
         "strategy": "CompoundPoissonSafetyStock",
         "trigger": "OrderUpToTrigger",
         "param_grid": {"target_service_level": [0.90, 0.95, 0.98]},
         "rationale": "Lumpy demand, hypothesis wants fewer stockouts -- distributional strategy fit directly to sparse+variable history."
       }
     ]
   }
   ```
   Every `param_grid` value must be a non-empty list (a single-value list is a
   valid one-point grid). Include every *required* constructor param --
   check `describe(strategy)` first if unsure.

   **`selection_rationale` is not optional filler.** It must state *why this
   subset* -- not the full registry -- was chosen, and it must actually
   mention one of `data_summary.demand_classes_present`. This is the audit
   trail for "why K strategies, not all of them," and step 4 hard-rejects a
   space that skips it.
4. **Validate before handing off**:
   ```
   python .claude/skills/propose-strategy-space/scripts/validate_strategy_space.py --strategy-space out.json
   ```
   Two hard rules on top of registry validity, enforced here (and re-checked
   by `log_decision.py` and `optimize-replenishment-cost`'s grid expansion --
   there's no path around them):
   - **At most half of the available safety-stock strategies** may appear as
     candidates (`len(list_safety_stock_strategies()) // 2` -- currently 5 of
     11). Never propose every strategy "to be safe"; a hypothesis implies a
     real subset, and if you can't narrow it, that's a sign to re-read the
     data/hypothesis, not to widen the candidate list.
   - **`selection_rationale` must be non-empty and grounded** in the actual
     `demand_classes_present` (see step 3).

   Also fix any reported unknown strategy/trigger/param or missing required
   param.
5. **Log the decision** (auditable record, one file per run):
   ```
   python .claude/skills/propose-strategy-space/scripts/log_decision.py \
     --strategy-space out.json --hypothesis "<the original hypothesis text>" --customer acme
   ```
   `--customer` is never inferred from a table/catalog name -- ask if it
   wasn't already given. Writes
   `experiments/strategy_space_decisions/acme/<timestamp>-<slug>.yaml`, and
   refuses to write anything if the strategy space fails step 4's validation
   -- an unaudited space can never end up looking logged/approved.
6. **Hand off** `out.json` to `optimize-replenishment-cost` (or let the
   `run-replenishment-strategy` orchestrator do steps 1-5 automatically as
   part of a full run).

## Key rules

- Never invent a strategy or param name -- `describe(name)` is the source of
  truth, and step 4 catches drift before it wastes a grid search.
- A hypothesis biases which candidates and params to try; it never excuses
  skipping classification in step 1.
- Only include a `NullSafetyStockStrategy` candidate if you have a specific
  reason to compare against zero buffer -- `optimize-replenishment-cost`'s
  underlying sweep already adds one automatically once 2+ configs run.
- Keep each `param_grid` small (2-4 values per param): grid search is
  exponential in the number of varied params, and a hypothesis implies a
  *region* to search, not "try everything".
- Never propose the full strategy roster to hedge against being wrong --
  the subset cap in step 4 is a hard rule precisely so "just try everything"
  is not an available shortcut. A wide, unfocused search is not more
  rigorous than a narrow, justified one; it's less auditable.

## Evals

`evals/hypotheses.jsonl` has fixed hypothesis fixtures with deterministic
pass/fail rules (`must_include_any`/`must_not_include` strategy families).
After producing a strategy space for one of these fixtures' hypothesis text,
grade it:
```
python .claude/skills/propose-strategy-space/evals/run_eval.py \
  --hypothesis-id lumpy-minimize-stockouts --strategy-space out.json
```
No LLM grading -- these are hard checks against the real registry and the
Syntetos-Boylan table, so results are exact and reproducible run to run.
`run_eval.py` calls the same `validate()` step 4 does, so a space that
proposes too many strategies or skips a grounded `selection_rationale` fails
the eval on that alone, before `must_include_any`/`must_not_include` are
even checked.
