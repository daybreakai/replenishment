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

## Notebooks

Worked examples under [`notebooks/`](notebooks/): safety-stock variants,
calibration with held-out validation, fill-rate/service-level probability,
plotting, and a stress test.

## Package layout

- `policy.py` — `ReplenishmentPolicy`, the unified policy class
- `portfolio.py` — `Portfolio`/`PortfolioResult`, the multi-item front door
- `report.py` — typed pydantic report layer: `build_report()`,
  `policy_runs_from_portfolio()`
- `classification.py` — `classify_demand()`, Syntetos-Boylan demand-shape router
  (smooth / erratic / intermittent / lumpy)
- `strategies/safety_stock.py` — `SqrtHorizonSafetyStock`, `KRmseSafetyStock`,
  `KMaeSafetyStock`, `FillRateSafetyStock`, `FixedErrorSafetyStock`
- `strategies/distributional_safety_stock.py` — `KingsFormulaSafetyStock`,
  `CompoundPoissonSafetyStock`
- `strategies/multiplier.py` — `MultiplierSafetyStockStrategy`,
  `NullSafetyStockStrategy`
- `strategies/resolver.py` — `resolve_safety_stock_strategy()`, picks a
  safety-stock strategy from data availability instead of demand shape
- `strategies/order_trigger.py` — `OrderUpToTrigger`,
  `FlatForecastOrderUpToTrigger`, `ReorderPointTrigger`
- `strategies/demand_buffer.py` — trend-chasing demand-buffer decorator
- `simulation.py` — `simulate_replenishment`, day-by-day lost-sales simulator
- `calibration.py` — grid-search calibration with train/validation split
- `io_.py` — data loaders and policy-construction helpers
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
