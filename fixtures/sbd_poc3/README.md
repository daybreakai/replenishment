# SBD POC3 local fixtures

Drop SBD's flat files here once Sebastian shares them (blocked on the
`data-exchange-demo-poc3-inbound` UC grant as of 2026-08-25 — see
#central-it-support). Nothing in this repo depends on that grant landing;
this folder lets us work entirely off local copies instead.

Drop one already-combined file here: `sbd_poc3.parquet` (or `.csv` — the
notebook picks either up). One dataframe, already joined across whatever
source tables (demand actual + inventory policy), in the repo's own
column shape:

- `unique_id`, `ds`, `demand`, `forecast`, `lead_time`,
  `holding_cost_per_unit`, `stockout_cost_per_unit`, `order_cost_per_order`,
  `current_stock`

Not this repo's job to do that join or fix source data quality (unit
mixing in lead_time_days, the 4-vocabulary product-id crosswalk, etc.) —
that's upstream, on whoever hands us the file. This folder just needs the
one file in the shape above.

Gitignored — see `.gitignore` (`fixtures/**/*.csv` and `fixtures/**/*.parquet`).
Never commit real customer data.
