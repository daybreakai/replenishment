---
name: run-replenishment-backtest
description: Run a replenishment safety-stock strategy + order trigger, with given hyperparameters, as a backtest across every unique_id in a dataset and report fill_rate/cost. Usage: /run-replenishment-backtest --strategy KRmseSafetyStock --params factor=1.65 --trigger OrderUpToTrigger --data fixtures/sbd_poc3/sbd_poc3.parquet
allowed-tools: Bash, Read, Glob
---

# Run a replenishment backtest

Turns "configure a strategy and see how it performs" into one command instead
of hand-rolling the per-`unique_id` loop every notebook in `notebooks/`
currently reinvents. See the repo root `CLAUDE.md` for the underlying
build -> wire -> simulate -> read pattern this skill is a runbook around, and
`references/strategy-selection.md` if the strategy family hasn't been picked
yet.

## Constants

- Script: `scripts/backtest.py` (this skill directory)
- Real-shaped fixture: `fixtures/sbd_poc3/sbd_poc3.parquet` (see its own
  `README.md` for provenance/columns)
- Example notebooks (pre-registry, hand-rolled — for reference only, don't
  copy their pattern): `notebooks/`
- Strategy/trigger discovery: `replenishment.strategies.list_safety_stock_strategies()`,
  `list_order_triggers()`, `describe(name)`
- Results log: `experiments/results/<customer>/results.jsonl` (repo root,
  git-committed) — append-only, one JSON line per run, partitioned by
  `--customer` (omit it for `experiments/results/_unscoped/results.jsonl`,
  e.g. a synthetic sanity check not tied to a real client). Same partitioning
  `propose-strategy-space`'s `experiments/strategy_space_decisions/<customer>/`
  and `segment-inventory-items`'s `experiments/segmentation_runs/<customer>/`
  use. `backtest.py` auto-warns when a run repeats an already-logged
  (strategy, params, trigger, data) combo within that customer's log. Run
  `scripts/history.py --customer <customer>` to browse past runs, or add
  `--diff <strategy_a> <strategy_b>` to compare two strategies' latest logged
  runs directly (config + fill_rate/total_cost deltas).

## Execution steps

1. **Resolve inputs.** From the user/conversation, get: `--strategy` (a name
   from `list_safety_stock_strategies()`), `--params` (its hyperparameters —
   run `describe(strategy)` first if unsure what it takes), `--trigger` (a
   name from `list_order_triggers()`, default `OrderUpToTrigger`), `--data`
   (`synthetic` for a fast sanity check, or a `.csv`/`.parquet` path), and
   `--horizon` if its default (1) isn't right. If the user hasn't named a
   strategy family yet, point them at `references/strategy-selection.md`.
   For MOQ/lead_time/review_period, see step 5 — these are dataset
   properties (often already columns in the data), not per-run knobs to
   guess a scalar for.
2. **Run the script** via Bash:
   ```
   python .claude/skills/run-replenishment-backtest/scripts/backtest.py \
     --strategy KRmseSafetyStock --params factor=1.65 --trigger OrderUpToTrigger \
     --data synthetic
   ```
   For a real CSV/parquet source whose columns don't already match
   `StandardSimulationRow` (see `CLAUDE.md`), pass `--actuals-field`/
   `--initial-on-hand-field` to point at the right columns instead of
   renaming the data.
3. **Relay the printed table as-is** — mean/median `fill_rate`, total
   `total_cost`, and a worst-5-by-`fill_rate` breakout, plus a cost breakdown
   (`holding_cost`/`stockout_cost`/`ordering_cost` — not just their sum) and
   inventory `turnover`/order count — two strategies can reach the same
   `total_cost` by very different means, and that only shows up in the
   breakdown. No file writes beyond the results log (below); if items
   failed, the script lists the first few error messages under the table.
4. **Sweep multiple configs instead of one.** Pass `--configs` (a path to a
   JSON file or an inline JSON array) instead of `--strategy`/`--params`/
   `--trigger`:
   ```
   python .claude/skills/run-replenishment-backtest/scripts/backtest.py \
     --data synthetic \
     --configs '[{"strategy": "KRmseSafetyStock", "params": {"factor": 1.65}},
                 {"strategy": "SqrtHorizonSafetyStock", "params": {"factor": 1.65}}]'
   ```
   All configs run against the same loaded dataset. Runs with 2+ configs
   automatically add a `NullSafetyStockStrategy` (zero buffer) baseline row
   per distinct trigger used, and rank variants against it by `total_cost`
   with `fill_rate`/`total_cost` deltas shown — pass `--no-baseline` to skip
   it. Every run (single or sweep) appends a line to that customer's results
   log (pass `--customer`, same as `propose-strategy-space`/
   `segment-inventory-items` — never inferred, ask if not given); pass
   `--no-log` for a throwaway run, or `--log-path` to redirect it entirely.
5. **MOQ/lead_time/review_period are dataset properties, not sweep axes.**
   These vary per SKU and are usually already columns in the data — read
   them from a column instead of guessing one scalar for the whole run:
   ```
   python .claude/skills/run-replenishment-backtest/scripts/backtest.py \
     --strategy KRmseSafetyStock --params factor=1.65 --data mydata.csv \
     --moq-field moq --review-period-field review_period \
     --lead-time-field supplier_lead_time
   ```
   `--moq-field`/`--review-period-field`/`--lead-time-field` are **global**
   (resolved once, shared by every config in a `--configs` sweep) and
   mutually exclusive with their scalar sibling (`--moq`, `--review-period`,
   `--lead-time`) — comparing strategies under different MOQs in the same
   sweep would confound the comparison. `--data synthetic` doesn't support
   `-field` flags (no arbitrary columns); a missing column errors immediately
   naming the columns that do exist — no silent fallback either way.
6. **`forecast_field` — which forecast to feed a policy — IS a sweep axis**,
   set per `--configs` entry (not global), since comparing forecast sources
   is a legitimate thing to test:
   ```
   --configs '[{"strategy": "NullSafetyStockStrategy", "forecast_field": "p90"},
               {"strategy": "KRmseSafetyStock", "params": {"factor": 1.65}}]'
   ```
   Defaults to `"forecast"` (the base column) when omitted; any other value
   is looked up in that row's `forecast_percentiles` (e.g. `"p50"`, `"p90"`),
   erroring immediately if the dataset doesn't have it. Pairing a percentile
   `forecast_field` with a non-`NullSafetyStockStrategy` prints an
   informational note (possible double-counted buffer) but still runs —
   nothing is forced or refused.

## Error handling

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: ... has no param(s) [...]` | Typo'd or nonexistent kwarg in `--params` | Run `describe(strategy)` to see valid params |
| `ValueError: ... requires [...]` | Missing a required kwarg (e.g. `FillRateSafetyStock` needs `target_fill_rate`) | Add it to `--params` |
| Per-item error `... requires actuals ...` | Strategy needs demand history but the data has none for that item | Check `CLAUDE.md`'s actuals-requirement list; verify `--actuals-field` points at real history |
| Per-item `SafetyStockRangeError` | `FillRateSafetyStock`'s target/mean/std combo has no exact solution in range | Pass `on_out_of_range=clip` in `--params` |
| Per-item `NegativeBinomialRangeError` | Item's demand isn't overdispersed (variance <= mean) | Pass `on_underdispersion=zero` in `--params`, or pick a different strategy for that item |
| `Unsupported --data ...` | Not `synthetic`, `.csv`, or `.parquet` | Convert first, or add the format if genuinely needed |
| `--configs is mutually exclusive with --strategy/--params/--trigger` | Both a sweep and a singular flag were given | Pick one mode |
| `--moq and --moq-field are mutually exclusive` (same for review-period/lead-time) | Both the scalar and `-field` flag were given for the same parameter | Pick one |
| `... is not supported with --data synthetic` | A `-field` flag was used with synthetic data (no arbitrary columns exist) | Use the matching scalar flag, or point `--data` at a real CSV/parquet |
| `... column '...' not found in ...; available columns: [...]` | A `-field` flag named a column that doesn't exist | Fix the column name from the list shown |
| `forecast_field '...' not found in forecast_percentiles; available: [...]` | A `--configs` entry's `forecast_field` isn't a real percentile key in the data | Fix it to one of the available keys, or `"forecast"` |

## Key rules

- Don't route through `Portfolio`/`io_.py`'s narrow string-keyed sweep API for
  new strategy work — it only covers 6 of ~11 strategies with one scalar
  param each. This skill (and `CLAUDE.md`) always construct the strategy
  class directly.
- This is a **lost-sales backtest**, not a live ordering decision —
  `simulation.py`'s own docstring: unmet demand is recorded but never
  backfilled once stock arrives.
- Always report the worst-N items, not just the aggregate — a good mean
  fill_rate can hide a handful of items in freefall.
- MOQ/lead_time/review_period describe the dataset (global, same across every
  config in a sweep); `forecast_field` describes the experiment (per-config).
  Don't confuse the two when designing a new comparison — varying MOQ between
  configs would confound whatever you're actually trying to test.
- In sweep mode, always compare a variant against the automatic baseline
  row, not just against other variants — a strategy can look best of three
  bad options and still lose to zero safety stock.
- `backtest.py` already warns on an exact-repeat config; use `scripts/history.py
  --diff <a> <b>` to compare two strategies deliberately instead of eyeballing
  two separate report tables.
