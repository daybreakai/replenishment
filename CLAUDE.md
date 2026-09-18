# replenishment — agent guide

Pure-Python (pandas/matplotlib only) inventory replenishment simulation library:
pick a safety-stock strategy and an order trigger, compose them into a
`ReplenishmentPolicy`, run `simulate_replenishment`, read the summary.

## Pattern: build -> wire -> simulate -> read

```python
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import KRmseSafetyStock
from replenishment.policy import ReplenishmentPolicy
from replenishment.simulation import simulate_replenishment

demand = [...]  # list[int], one value per period
forecast = TimeSeries.from_values([20.0] * 90)
actuals = TimeSeries.from_values(demand)

policy = ReplenishmentPolicy.order_up_to(   # or .reorder_point(...)
    forecast=forecast,
    actuals=actuals,                         # required only if the strategy needs it, see below
    safety_stock=KRmseSafetyStock(factor=1.65),
    lead_time=3, review_period=7, forecast_horizon=7,
)

result = simulate_replenishment(
    policy=policy, demand=demand, periods=90, initial_on_hand=100,
    lead_time=3, holding_cost_per_unit=1.0, stockout_cost_per_unit=5.0,
)

result.summary.fill_rate, result.summary.total_cost, result.summary.avg_on_hand
```

`.order_up_to(...)` pins `trigger=OrderUpToTrigger()`; `.reorder_point(...)` pins
`trigger=ReorderPointTrigger()`. Construct `ReplenishmentPolicy(...)` directly
only to use a different trigger (`FlatForecastOrderUpToTrigger`,
`FlatReorderPointTrigger` — flat-forecast variants for rolling one-step-ahead CV
series, see their docstrings in `strategies/order_trigger.py`).

## Discover strategies and hyperparameters

```python
from replenishment.strategies import list_safety_stock_strategies, list_order_triggers, describe

list_safety_stock_strategies()   # -> ['CompoundPoissonSafetyStock', 'DemandBufferDecorator', ...]
list_order_triggers()            # -> ['FlatForecastOrderUpToTrigger', 'FlatReorderPointTrigger', ...]
describe("KRmseSafetyStock")     # -> {"name", "doc", "params": {"factor": {...}, "k": {...}}}
```

`describe()` reads the actual `__init__` signature, so alias kwargs (`factor=`/`k=`
on `SqrtHorizonSafetyStock`/`KRmseSafetyStock`/`KMaeSafetyStock`) show up correctly.
It is discovery only: construct the class yourself and pass it as
`safety_stock=`/`trigger=` on `ReplenishmentPolicy` — nothing here wraps or hides
the classes (`src/replenishment/strategies/registry.py`).

**Actuals requirement**: `SqrtHorizonSafetyStock`, `KRmseSafetyStock`,
`KMaeSafetyStock`, `FillRateSafetyStock`, `KingsFormulaSafetyStock`,
`CompoundPoissonSafetyStock`, `NegativeBinomialSafetyStock` all need `actuals=` on
the policy (forecast error or raw demand history). `MultiplierSafetyStockStrategy`/
`NullSafetyStockStrategy` need neither. `DemandBufferDecorator(wrapped=..., strength=...)`
needs actuals only if `wrapped` does. `ReplenishmentPolicy.__post_init__` raises
immediately if `actuals` is missing when required.

## Data contract: StandardSimulationRow (io_.py)

Loading a portfolio of rows from CSV/DataFrame instead of hand-building one
`TimeSeries`? The shape is `replenishment.io_.StandardSimulationRow`:

```
unique_id: str, ds: str, demand: int, forecast: int,
actuals: int | float | None,
holding_cost_per_unit / stockout_cost_per_unit / order_cost_per_order: float,
lead_time: int, initial_on_hand / current_stock: int,
forecast_percentiles: Mapping[str, int], is_forecast: bool = False
```

Read/write via `io_.iter_standard_simulation_rows_from_csv`,
`io_.standard_simulation_rows_from_dataframe`,
`io_.standard_simulation_rows_to_dataframe`, `io_.write_standard_simulation_rows_to_csv`.

For a `.csv`/`.parquet` **path** specifically (not a DataFrame you already
have in memory), prefer `io_.load_standard_simulation_rows(path,
actuals_field=..., initial_on_hand_field=..., lead_time_field=...)` over
calling the two functions above directly -- it's the single place that
branches on file extension and threads through column-name overrides, used
by both the `run-replenishment-backtest` and `propose-strategy-space`
skills so their `--actuals-field`/`--initial-on-hand-field`/
`--lead-time-field` flags can't drift out of sync with each other.

## What NOT to use for new strategy work

`io_.py`'s `_SAFETY_STOCK_STRATEGIES`/`_FACTOR_KWARG` and `Portfolio`
(`portfolio.py`) are a narrow, legacy, string-keyed sweep path (6 of ~11
strategies, one scalar kwarg each). Don't route new strategy/hyperparameter work
through them — construct the class directly (above) and build a
`ReplenishmentPolicy` yourself.

## Tests

`pytest` from repo root (`testpaths = ["tests"]`). Strategy tests live under
`tests/strategies/`.

## Past backtest results

`experiments/results/<customer>/results.jsonl` (repo root, git-committed;
`experiments/results/_unscoped/results.jsonl` when no `--customer` was
given) is an append-only log of every `run-replenishment-backtest`/
`optimize-replenishment-cost` invocation (single config, sweep, or grid
search) — one JSON line per run with a timestamp, the data source, and each
row's strategy/params/trigger/fill_rate/total_cost. Same per-customer
partitioning `propose-strategy-space`'s `experiments/strategy_space_decisions/`
and `segment-inventory-items`'s `experiments/segmentation_runs/` use. Read a
customer's log directly or run
`.claude/skills/run-replenishment-backtest/scripts/history.py --customer
<customer>` before re-running an expensive sweep that may already have been
tried.
