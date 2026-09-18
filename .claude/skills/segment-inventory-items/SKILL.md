---
name: segment-inventory-items
description: Composable inventory segmentation -- assign each item's timeseries one or more segment labels by running any registered strategy (demand-shape ADI/CV2, revenue-tier ABC, variability-tier XYZ, their matrix, or a hierarchy-rolled-up variant) against a client's real Databricks tables, then append-write labeled segments to that client's own t_segmentation table. New strategies plug into the same registry without changing the pipeline. Usage: /segment-inventory-items --catalog acme_prod_sc --schema data_store --profile <profile> --customer acme --strategies adi_cv2 abc_revenue
allowed-tools: Bash, Read, Glob
---

# Segment inventory items

The core idea: one item's timeseries in, one or more composable segment
labels out. Each label comes from a *strategy* -- a named entry in
`SEGMENTATION_STRATEGIES` (`scripts/segmentation_strategies.py`) that
declares which tables it needs and how it turns raw rows into a label. The
pipeline itself (pull -> validate -> compute -> audit -> write-back) never
changes; adding a new way to segment items means adding a new registry
entry, not a new script. Today's entries cover demand-shape (`adi_cv2`,
grounded in `t_outbound_shipment`) and revenue/variability (`abc_revenue`,
`xyz_variability`, `abc_xyz_matrix`, `abc_revenue_by_hierarchy`, grounded in
`t_outbound_shipment`+`t_product`) -- see
`references/extra-data-requirements.md` for the shape of a strategy that
needs more than that. Segment labels are also a signal `propose-strategy-space`
can use for *which* items to run a given hypothesis against, but this skill
stands alone as a reporting/classification step too.

**`adi_cv2` vs. `propose-strategy-space`'s own classification**: both wrap
the same `classify_demand()`, but `propose-strategy-space/scripts/classify_items.py`
classifies fresh from a local `--data` CSV/parquet every run, while this
skill's `adi_cv2` classifies from a customer's *live* Databricks
`t_outbound_shipment` and persists the result to `t_segmentation`. They're
independent by design (different sources, different lifetimes) -- don't
treat one as a cache of the other.

## Constants

- Strategy registry (source of truth for required tables): `scripts/segmentation_strategies.py`
- Validation script (hard rule): `scripts/validate_segmentation.py`
- SQL builder + audit-log writer: `scripts/build_segmentation_sql.py`
- Orchestrator: `scripts/run_segmentation.py`
- Extra-data rationale: `references/extra-data-requirements.md`
- Audit log: `experiments/segmentation_runs/<customer>/<timestamp>.sql` (repo
  root, one file per run -- logs the exact SQL that ran, both the pulls and
  the write-back)
- Write-back target: `<catalog>.<schema>.t_segmentation` (append-only)

## Default data pull

Every run always pulls `t_outbound_shipment` (demand signal) and `t_product`
(for `unit_cost`, the default price source) -- see `DEFAULT_TABLES` in
`run_segmentation.py`. If `t_product.unit_cost` is null for any item feeding
a revenue-based strategy, `compute_segments` raises rather than falling back
to another price source; ask the user how to handle those items, don't
silently substitute.

A strategy needing more than the default two tables (currently only
`abc_revenue_by_hierarchy`, which needs `t_product_hierarchy`) must have its
extra table passed explicitly via `--tables`, or `validate_segmentation.py`
hard-rejects the request. See `references/extra-data-requirements.md` for
why each strategy needs what it needs.

## Execution steps

1. **Confirm `--catalog`, `--schema`, `--profile`, `--customer` with the
   user** -- never infer any of these (catalog naming varies per client,
   e.g. `pourri_prod_sc`, `hasbro_prod_sc`; profile selection must never be
   auto-chosen per `databricks-core`'s own rule).
2. **Pick strategies** from `segmentation_strategies.SEGMENTATION_STRATEGIES`.
   Running `adi_cv2` alone needs no price data; anything revenue-based
   (`abc_revenue`, `xyz_variability`, `abc_xyz_matrix`, `abc_revenue_by_hierarchy`)
   needs `t_product` too (already in the default pull) plus, for the
   hierarchy variant, `t_product_hierarchy` via `--tables`.
3. **Dry-run first against a real client catalog you haven't segmented
   before**:
   ```
   python .claude/skills/segment-inventory-items/scripts/run_segmentation.py \
     --catalog acme_prod_sc --schema data_store --profile <profile> \
     --customer acme --strategies adi_cv2 abc_revenue --dry-run
   ```
   `--dry-run` still queries the source tables and computes segments (so you
   can review the numbers), still writes the audit SQL file, but skips
   executing the `CREATE TABLE`/`INSERT` against the client's catalog.
4. **Run for real** (drop `--dry-run`) once the dry-run output looks right.
   This is append-only: `CREATE TABLE IF NOT EXISTS
   <catalog>.<schema>.t_segmentation` followed by one `INSERT`, tagged with a
   `run_timestamp`. There is no `DROP`/`MERGE`/`DELETE`/overwrite path in
   this skill -- rerunning never destroys a prior run's rows.
5. **Hand off** the printed records (or query `t_segmentation` directly) to
   `propose-strategy-space` as an extra signal on which items share a demand
   shape or revenue tier, if that's the next step.

## Key rules

- `t_outbound_shipment` is non-negotiable -- every strategy in the registry
  requires it (`test_registry_entries_all_require_outbound_shipment` in
  `test_segmentation_strategies.py` enforces this doesn't silently drift).
- Never add a table requirement to an existing strategy's entry in
  `SEGMENTATION_STRATEGIES` to work around a validation failure -- if a
  strategy needs new data, give it a new name (see the promo-adjusted
  example in `references/extra-data-requirements.md`) so the audit trail on
  what ran against what data stays honest.
- `validate_segmentation.py`'s check is a hard rule, same spirit as
  `propose-strategy-space`'s subset cap: there's no flag to bypass a missing
  required table.
- The write-back schema (`_SEGMENTATION_TABLE_COLUMNS` in
  `build_segmentation_sql.py`) is deliberately thin -- class/tier labels and
  total revenue, not raw ADI/CV² numbers `classify_demand()` doesn't expose
  anyway. Don't invent extra numeric columns without a real consumer for them.
