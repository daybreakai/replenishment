---
name: optimize-replenishment-cost
description: Load real MOQ/review-period/lead-time/cost data from a Databricks table, grid-search a propose-strategy-space output against it, and report the min-total_cost config that still clears given constraints (e.g. a fill-rate floor). Usage: /optimize-replenishment-cost --table catalog.schema.table --profile <p> --strategy-space out.json --min-fill-rate 0.95
allowed-tools: Bash, Read, Glob
---

# Optimize replenishment cost under constraints

Turns a `propose-strategy-space` JSON output into an actual cost-minimizing
config: expand every candidate's `param_grid` into concrete configs, run them
all, keep only the ones that clear the given constraints, and report the
cheapest survivor. No new simulation engine -- `grid_search.py` reuses
`run-replenishment-backtest/scripts/backtest.py`'s `_run_one_config`/
`_aggregate`/`_load_rows` directly.

## Constants

- Databricks loader: `scripts/load_from_databricks.py`
- Grid search + constraint filter: `scripts/grid_search.py`
- Underlying sweep engine (reused, not duplicated):
  `../run-replenishment-backtest/scripts/backtest.py`
- Results log: `experiments/results/<customer>/results.jsonl` (repo root)
  -- grid-search runs are tagged `"source": "grid_search"` in that log, same
  per-customer file `run-replenishment-backtest` writes to (`--customer`,
  never inferred; omit for `experiments/results/_unscoped/`).

## Execution steps

1. **Load the real item-master + demand data from Databricks**, if you're not
   already working from a local CSV/parquet. Ask the user which Databricks
   CLI profile to use -- never default or guess it:
   ```
   python .claude/skills/optimize-replenishment-cost/scripts/load_from_databricks.py \
     --table daybreakpoc_agent_playground.replenishment_sandbox.standard_simulation_rows \
     --profile jack.rodenberg@daybreak.ai --out data.csv
   ```
   The table must already be `StandardSimulationRow`-shaped (see repo root
   `CLAUDE.md`) plus whatever extra columns you'll point `--moq-field`/
   `--review-period-field`/`--lead-time-field` at in step 3 -- MOQ and review
   period come from the data model, never a guessed scalar (see
   `daybreakpoc_agent_playground.replenishment_sandbox.standard_simulation_rows`
   for the reference shape: `moq`, `review_period`, `lead_time` columns).
   `--where` narrows the query (e.g. `--where "site_id = 'US01'"`).
2. **Get a strategy space** from the `propose-strategy-space` skill first if
   you don't have one yet -- `grid_search.py` validates it the same way that
   skill's own validator does, and will reject a hand-written one with the
   same errors.
3. **Run the grid search**:
   ```
   python .claude/skills/optimize-replenishment-cost/scripts/grid_search.py \
     --strategy-space out.json --data data.csv --customer acme \
     --moq-field moq --review-period-field review_period --lead-time-field lead_time \
     --min-fill-rate 0.95
   ```
   `--min-fill-rate` (mean `fill_rate` floor) and `--max-total-cost` (hard
   budget cap) are the two constraint knobs; combine them if both apply. Every
   other flag `backtest.py` supports also works here unchanged (`--horizon`,
   `--holding-cost`/`--stockout-cost`/`--order-cost` overrides, `--data
   synthetic` for a fast sanity check before spending a real Databricks
   query, `--no-log`).
4. **Relay the printed table as-is**: N configs tried, N survived, then the
   ranked survivor table (fill_rate, total_cost, and the holding/stockout/
   ordering breakdown -- not just the total) and the named best config. If
   zero configs survive, that's a real result -- report it, then either
   loosen the constraint or ask `propose-strategy-space` to widen its
   `param_grid`s. Don't silently pick the least-bad violator.
5. **A large strategy space is slow, not wrong** -- grid size is the product
   of every candidate's `param_grid` cardinalities. If a run is taking too
   long, that's a signal to go back to `propose-strategy-space` and narrow
   the grids, not to change the search here.

## Error handling

| Symptom | Cause | Fix |
|---|---|---|
| `strategy space failed validation: ...` | The strategy space has an unknown strategy/trigger/param, or a missing required param | Fix the JSON, or re-run `propose-strategy-space`'s validator for the exact same errors |
| `strategy space produced zero configs` | `candidates` is empty | Get a non-empty strategy space from `propose-strategy-space` |
| `No config satisfied the constraints` | Every candidate's fill_rate/cost missed the floor/cap | Loosen `--min-fill-rate`/`--max-total-cost`, or widen `param_grid` values |
| `databricks query failed: ...` | Bad table name, wrong profile, or no warehouse access | Check `--table`/`--profile`; see `databricks-core` skill for auth/profile issues |
| Any `backtest.py`-native error (unknown param, missing actuals, etc.) | Same causes as in `run-replenishment-backtest`'s own error table | See that skill's `SKILL.md` error table |

## Key rules

- MOQ/review_period/lead_time are dataset properties (per `-field` flag,
  resolved once, shared across the whole grid) -- never a scalar you invent,
  and never varied as a grid axis themselves (that would confound the
  strategy comparison). This mirrors `run-replenishment-backtest`'s own rule.
- Always report the full survivor table, not just the single best row --
  ties and near-ties matter when picking a config a human will actually ship.
- `grid_search.py` logs to the same per-customer results log as
  `run-replenishment-backtest` (pass `--customer`); check that log (or its
  `history.py --customer <customer>`) before re-running an expensive sweep
  against the same table.
