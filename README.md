# replenishment

Strategy-composed inventory replenishment library: one `ReplenishmentPolicy`
class built from a forecast `TimeSeries`, a swappable `SafetyStockStrategy`,
and an `OrderTrigger` (order-up-to or reorder-point) — instead of picking
among several near-duplicate policy classes.

## Install

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import KRmseSafetyStock
from replenishment.policy import ReplenishmentPolicy
from replenishment.simulation import simulate_replenishment

rng = np.random.default_rng(0)
demand = np.maximum(0, rng.normal(20, 5, 90)).round().astype(int)

forecast = TimeSeries(values=[20.0] * 90)
actuals = TimeSeries(values=demand.tolist())

policy = ReplenishmentPolicy.order_up_to(
    forecast=forecast,
    actuals=actuals,
    safety_stock=KRmseSafetyStock(k=1.65),
    lead_time=3,
    review_period=7,
    forecast_horizon=7,
)

result = simulate_replenishment(
    policy=policy,
    demand=demand.tolist(),
    periods=90,
    initial_on_hand=100,
    lead_time=3,
    holding_cost_per_unit=1.0,
    stockout_cost_per_unit=5.0,
)

print("fill_rate:", result.summary.fill_rate)
print("total_cost:", result.summary.total_cost)
print("avg on-hand:", result.summary.avg_on_hand)
```

Swap `safety_stock=` for `FillRateSafetyStock(target_fill_rate=0.95)` or
`SqrtHorizonSafetyStock(k=...)`, or use `.reorder_point(...)` instead of
`.order_up_to(...)` — same call shape.

## Panel usage (many items at once)

Real inputs are usually a flat item/date/forecast/demand + cost table, not one
series at a time. `io_.py` builds `ReplenishmentPolicy` objects straight from
that shape:

```python
from replenishment.io_ import (
    generate_standard_simulation_rows,
    standard_simulation_rows_to_dataframe,
    standard_simulation_rows_from_dataframe,
    build_point_forecast_article_configs_from_standard_rows,
)

# swap this generator for your own panel df with the same columns:
# unique_id, ds, demand, forecast, actuals, holding_cost_per_unit,
# stockout_cost_per_unit, order_cost_per_order, lead_time, current_stock
rows = generate_standard_simulation_rows(n_unique_ids=3, periods=60, seed=0)
df = standard_simulation_rows_to_dataframe(rows)

configs = build_point_forecast_article_configs_from_standard_rows(
    standard_simulation_rows_from_dataframe(df),
    service_level_factor=1.65,
    safety_stock_method="k_rmse",
)

for unique_id, config in configs.items():
    result = config.simulate()
    print(unique_id, "fill_rate:", result.summary.fill_rate, "total_cost:", result.summary.total_cost)
```

`build_point_forecast_article_configs_from_standard_rows` groups rows by
`unique_id`, builds one policy per item, and returns an
`ArticleSimulationConfig` you call `.simulate()` on. Also accepts
`policy_mode="rop"` for reorder-point instead of order-up-to, and per-item
overrides (dict keyed by `unique_id`) for any of the cost/factor args.

## Notebooks

Worked examples under [`notebooks/`](notebooks/): safety-stock variants,
calibration with held-out validation, fill-rate/service-level probability,
plotting, and a stress test.

## Package layout

- `policy.py` — `ReplenishmentPolicy`, the unified policy class
- `strategies/safety_stock.py` — `SqrtHorizonSafetyStock`, `KRmseSafetyStock`,
  `KMaeSafetyStock`, `FillRateSafetyStock`
- `strategies/order_trigger.py` — `OrderUpToTrigger`, `ReorderPointTrigger`
- `strategies/demand_buffer.py` — trend-chasing demand-buffer decorator
- `simulation.py` — `simulate_replenishment`, day-by-day lost-sales simulator
- `calibration.py` — grid-search calibration with train/validation split
- `io_.py` — data loaders and policy-construction helpers
- `segmentation.py` (pluggable `SegmentRule`s: ABC, XYZ, demand-pattern,
  attribute-tag, explicit group, composed into one `SegmentKey` per item)
- `segment_policy.py` (`SegmentPolicyMap`: segment id to policy knobs,
  YAML-loadable default plus a mergeable runtime override map)
- `viz.py` — plotting
- `timeseries.py` — `TimeSeries`, the shared forecast/actuals abstraction

## Notes

- Lost-sales model: unmet demand is recorded but never backfilled once new
  stock arrives.
- `FillRateSafetyStock` raises `SafetyStockRangeError` by default when a
  fill-rate target falls outside the solvable range, instead of silently
  clipping — pass `on_out_of_range="clip"` for the old behavior.
- `calibration.optimize()` scores candidates on a held-out validation slice
  distinct from the search slice.
