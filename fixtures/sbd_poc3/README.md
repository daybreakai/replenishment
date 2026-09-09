# sbd_poc3 fixture

`sbd_poc3.parquet` — a real-shaped client data extract (codename "SBD",
proof-of-concept round 3), used as a realistic-scale/realistic-noise
alternative to `generate_standard_simulation_rows(...)`'s synthetic data.
**Not committed to git** (see repo `.gitignore` — present locally only); if
this file is missing, `tests/test_notebooks.py`'s sbd_poc3 fixture test skips
rather than failing.

## Shape

- ~939,000 rows, 65,774 unique `unique_id`s (SKU-like codes, e.g. `"00 20 72
  V04 XS"`, `J7315M`, `50902CLCU3419`)
- Date range: `2024-05-01` to `2026-01-01`
- Columns: `unique_id, ds, demand, forecast, lead_time,
  holding_cost_per_unit, stockout_cost_per_unit, order_cost_per_order,
  current_stock`
- Real-world statistics, not synthetic: demand ranges from roughly -984,000
  to 9,559,940 with high variance (std ~35,000 vs mean ~1,903) — expect
  outliers, not a clean gaussian.

## Loading it

This fixture has **no `actuals`/`history` column and no
`initial_on_hand`/`initial_demand` column**, so
`standard_simulation_rows_from_dataframe(...)` needs field overrides to load
it (it otherwise assumes those columns exist under their default names):

```python
import pandas as pd
from replenishment.io_ import standard_simulation_rows_from_dataframe

df = pd.read_parquet("fixtures/sbd_poc3/sbd_poc3.parquet")  # requires pyarrow or fastparquet
rows = standard_simulation_rows_from_dataframe(
    df, actuals_field="demand", initial_on_hand_field="current_stock",
)
```

The `run-replenishment-backtest` skill's script exposes the same overrides as
`--actuals-field`/`--initial-on-hand-field`:

```
python .claude/skills/run-replenishment-backtest/scripts/backtest.py \
  --strategy KRmseSafetyStock --params factor=1.65 --trigger OrderUpToTrigger \
  --data fixtures/sbd_poc3/sbd_poc3.parquet \
  --actuals-field demand --initial-on-hand-field current_stock
```

`pyarrow` is a project dependency (added specifically to read this fixture),
so plain `pd.read_parquet(...)` works with no extra setup.

**Known data-quality caveat**: this is a real, messy extract, not a cleaned
dataset — some `unique_id`s have negative `demand` values, which can produce
nonsensical negative `fill_rate`/`NaN` `total_cost` for those specific items
in a backtest report. That's the fixture, not the simulation code; don't
"fix" it by patching `simulate_replenishment` — filter or flag those rows
before using them for anything beyond exercising the pipeline at realistic
scale.
