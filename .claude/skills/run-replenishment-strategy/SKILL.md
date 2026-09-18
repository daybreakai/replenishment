---
name: run-replenishment-strategy
description: End-to-end -- given a natural-language inventory hypothesis, a Databricks table, and cost/service constraints, load the data, propose a strategy space, grid-search it, and report the min-cost config that clears the constraints. Usage: /run-replenishment-strategy --hypothesis "..." --table catalog.schema.table --profile <p> --customer acme --min-fill-rate 0.95
allowed-tools: Bash, Read, Write, Glob
---

# Run the replenishment strategy pipeline end-to-end

Thin orchestrator chaining three skills. No new script -- each step is one
of the sibling skills' own scripts, run in order. Use this when the ask is
"just tell me what to run", and use the sibling skills directly when you only
need one stage (e.g. re-running the grid search with a tweaked constraint
against an already-produced strategy space).

**Optional pre-step -- segmentation.** If `--customer` doesn't have item
segment labels yet (check `<catalog>.<schema>.t_segmentation`) and the
hypothesis would benefit from one (e.g. "focus on our top-revenue items" ->
`abc_revenue`), run `../segment-inventory-items/SKILL.md` first and mention
the resulting classes/tiers when writing the hypothesis in step 2. This
pipeline doesn't require segmentation and never runs it automatically --
`propose-strategy-space`'s own classification (step 2) is sufficient on its
own for a hypothesis that doesn't reference revenue/ABC/XYZ tiers.

## Pipeline

1. **Load data** -- `../optimize-replenishment-cost/scripts/load_from_databricks.py`
   against the given `--table`/`--profile` (ask for the profile if not
   given; never default it). Produces a local CSV.
2. **Classify + propose a strategy space** -- follow
   `../propose-strategy-space/SKILL.md` end to end against that CSV and the
   given `--hypothesis`, including its final `log_decision.py` step with the
   given `--customer` (ask for it if not given; never infer it from the
   table/catalog name). Produces `strategy_space.json`, already validated
   and logged to `experiments/strategy_space_decisions/<customer>/`.
3. **Grid-search under constraints** -- follow
   `../optimize-replenishment-cost/SKILL.md`'s grid-search step (step 3
   onward) with that `strategy_space.json`, the loaded CSV, the same
   `--customer` from step 2, and whatever `--min-fill-rate`/`--max-total-cost`
   were given. Pass through any `--moq-field`/`--review-period-field`/
   `--lead-time-field` the table needs.
4. **Report**, in this exact shape:
   ```
   Hypothesis: <verbatim>
   Demand classes present: <from data_summary.demand_classes_present>
   Selection rationale: <selection_rationale from the strategy space>

   Candidates searched:
   - <strategy>(<params grid>) / <trigger> -- <rationale>
   - ...

   <the grid_search.py survivor table, relayed as-is>

   Best under constraints: <strategy>(<params>) / <trigger>
   Logged: experiments/strategy_space_decisions/<customer>/<file>
   ```
   If nothing survived the constraints, replace the last two lines with a
   one-line statement of that fact plus which knob to loosen first (the
   constraint, or the strategy space's `param_grid`s) -- never pick
   something that didn't actually qualify just to have an answer.

## Key rules

- Every sibling skill's own error table and key rules still apply here --
  this skill adds no new failure modes, only sequencing.
- Don't skip step 2's classification step just because the hypothesis names
  a demand shape already -- `propose-strategy-space` flags a mismatch
  between the claimed and the actual shape, and that's a real finding worth
  surfacing, not something to paper over.
- If the user only wants one stage re-run (e.g. "try a lower fill-rate
  floor"), don't re-run the whole pipeline -- go straight to
  `optimize-replenishment-cost` with the existing `strategy_space.json` and
  CSV.
