# Strategy-Composed Replenishment Engine — Design Spec

**Date:** 2026-08-24
**Status:** Approved for planning
**Origin:** Fork/rearchitect of `janrth/replenishment` into a new standalone repo, `daybreakai/replenishment`.

## 1. Motivation

`janrth/replenishment`'s `policies.py` implements 7 inventory-control policies (`ReorderPointPolicy`, `ForecastBasedPolicy`, `ForecastSeriesPolicy`, `PointForecastOptimizationPolicy`, `RopPointForecastOptimizationPolicy`, `LeadTimeForecastOptimizationPolicy`, `PercentileForecastOptimizationPolicy`, `RopPercentileForecastOptimizationPolicy`, `EmpiricalMultiplierPolicy`, `RopEmpiricalMultiplierPolicy`) as near-identical dataclasses. Each repeats the same `__post_init__` validation, the same `_forecast_value_for`/`_forecast_sum_for` lookup helpers, and the same demand-buffer-multiplier call. 1,446 lines for what is structurally ~150 lines of unique logic, copy-pasted ~7–10x.

This spec rearchitects the policy layer into one composable class built from three orthogonal strategies, while carrying the rest of the repo (`io.py`, `plotting.py`, `simulation.py`, `optimization.py`, `service_levels.py`) forward with adaptation rather than a rewrite. Two known weak points get fixed as part of this pass (not deferred):

1. `inverse_normal_loss` silently clips to its ±6σ search bound instead of signaling that a fill-rate target/demand combination fell outside the range it can solve.
2. `optimize_service_level_factors` (and its aggregation-window sibling) scores each candidate factor on the same window it's chosen from — no held-out check that the winning factor isn't just fit to that window's noise.

Explicitly out of scope: no Daybreak demand-shape classification (Syntetos-Boylan/King's/compound-Poisson) is ported in. This repo's safety-stock methods stay forecast-residual-based, as janrth's are today.

## 2. Architecture

```
math/          pure functions: normal_quantile, normal_pdf/cdf, normal_loss,
                inverse_normal_loss, rmse, mae. No classes.

strategies/     the 3 swappable axes:
                  - TimeSeries          (shared forecast/actuals abstraction)
                  - SafetyStockStrategy (sqrt_horizon, k_rmse, k_mae, fill_rate,
                                          multiplier, null; + demand-buffer decorator)
                  - OrderTrigger        (order_up_to, reorder_point)

policy/         ReplenishmentPolicy — the one unified class, composes one of each
                strategy above.

simulation/     InventoryState + day-by-day simulator. Ported from janrth's
                simulation.py; only touches ReplenishmentPolicy's public
                order_quantity_for(state), so this layer changes minimally.

calibration/    backtest + grid-search loop. Ported from janrth's optimization.py,
                adapted to score any ReplenishmentPolicy, with the train/validation
                split fix (see §4).

io/             data loaders. Ported from janrth's io.py, adapted to the new
                TimeSeries/ReplenishmentPolicy types.

viz/            plotting. Ported from janrth's plotting.py, adapted.
```

Dependency direction is strictly top-to-bottom: `policy` depends on `strategies`, never the reverse. `calibration`, `simulation`, `io`, `viz` depend on `policy`'s public interface only, never on strategy internals. This is what lets any strategy be tested, replaced, or added without touching the other layers.

## 3. Components

### `TimeSeries`
Wraps a list or callable; handles out-of-range lookup (extend-last-value) and missing-value skip during error computation. Replaces janrth's duplicated `_forecast_value_for`/`_forecast_sum_for`/`_actual_model` logic (currently reimplemented per policy class). Used for both forecast and actuals — a percentile forecast is still just a `TimeSeries`; there is no separate percentile type at this layer.

### `SafetyStockStrategy` (protocol)
One method: `compute(*, forecast: TimeSeries, actuals: TimeSeries | None, period: int, lead_time: int, horizon: int, service_level_factor: float, service_level_mode: str) -> float`.

Variants:
- `SqrtHorizonSafetyStock` — `z · error · √horizon` (janrth's default/legacy method).
- `KRmseSafetyStock`, `KMaeSafetyStock` — `z · error`, flat (no horizon scaling).
- `FillRateSafetyStock` — normal-loss inversion. Constructor param `on_out_of_range: Literal["raise", "clip"] = "raise"`. On `"raise"`, a fill-rate/demand combination that would clip to the ±6σ bound raises `SafetyStockRangeError` naming the target and the bound hit, instead of silently returning the boundary value.
- `MultiplierSafetyStockStrategy(multiplier: float)` — `safety_stock = forecast_qty · (multiplier − 1)`. Folds in `EmpiricalMultiplierPolicy`'s flat-multiplier behavior under the same interface as every other strategy (multiplier ≥ 1 → positive buffer; multiplier < 1 is rejected at construction, matching janrth's existing validation).
- `NullSafetyStockStrategy` — returns 0. Used when the forecast itself is already a target quantile (the old percentile-forecast policies); no separate policy class needed for that case.
- `DemandBufferDecorator(wrapped: SafetyStockStrategy, strength: float, reference: float | None, max_multiplier: float | None)` — wraps any strategy above, applies the trend-chasing uplift janrth calls `demand_buffer_strength` today. A decorator, not a parameter every strategy has to remember to read.

### `OrderTrigger` (protocol)
One method: `order_quantity(*, state: InventoryState, forecast_qty: float, safety_stock: float, review_period: int) -> int`.

Variants:
- `OrderUpToTrigger` — periodic review; order up to `forecast_qty + safety_stock`. Covers `ForecastBasedPolicy`, `PointForecastOptimizationPolicy`, `LeadTimeForecastOptimizationPolicy`, `ForecastSeriesPolicy`, `EmpiricalMultiplierPolicy`.
- `ReorderPointTrigger` — continuous review; order only when `inventory_position ≤ reorder_point`, up to `reorder_point + cycle_stock`. Covers `RopPointForecastOptimizationPolicy`, `RopPercentileForecastOptimizationPolicy`, `RopEmpiricalMultiplierPolicy`.

### `ReplenishmentPolicy`
The one unified class.

```python
class ReplenishmentPolicy:
    def __init__(
        self,
        *,
        forecast: TimeSeries,
        actuals: TimeSeries | None = None,
        lead_time: int = 0,
        review_period: int = 1,
        forecast_horizon: int = 1,
        safety_stock: SafetyStockStrategy,
        trigger: OrderTrigger,
    ) -> None: ...

    def order_quantity_for(self, state: InventoryState) -> int: ...

    @classmethod
    def order_up_to(cls, *, forecast, actuals=None, safety_stock, lead_time=0,
                     review_period=1, forecast_horizon=1) -> "ReplenishmentPolicy": ...

    @classmethod
    def reorder_point(cls, *, forecast, actuals=None, safety_stock, lead_time=0,
                       review_period=1, forecast_horizon=1) -> "ReplenishmentPolicy": ...
```

Construction-time validation (`lead_time ≥ 0`, `review_period > 0`, `forecast_horizon > 0`, and — new — `actuals is not None` whenever `safety_stock` needs it) happens once here and in the strategy constructors, not seven times.

**Net class count**: 7 policy classes → 1 policy class + 2 triggers + 6 safety-stock strategies (5 real + null) + 1 decorator + 1 shared `TimeSeries`. Every piece above is independently unit-testable in isolation.

## 4. Data Flow

```
historical rows
      │  split(backtest, forecast/eval)
      ▼
backtest window ──► calibration.optimize(strategy_family, candidate_params)
                     grid-searches params; each candidate is scored by running
                     simulation.run() on a TRAIN slice of the backtest window
                     and evaluated on a held-out VALIDATION slice — the fix for
                     the overfit issue (today's optimize_service_level_factors
                     scores and picks on the same window)
      │  best params
      ▼
ReplenishmentPolicy(forecast, actuals, safety_stock=<chosen strategy>, trigger=<chosen>)
      │
      ▼
eval/forecast window ──► simulation.run(policy)
                          day-by-day: policy.order_quantity_for(state) each period,
                          InventoryState updates (on-hand, position, cost)
      │
      ▼
results ──► io.to_dataframe() / viz.plot()  (same decision charts as today)
```

## 5. Error Handling

- **Construction-time validation, once.** Moves out of 7 duplicated `__post_init__`s into `ReplenishmentPolicy.__init__` and the individual strategy constructors.
- **`FillRateSafetyStock` raises by default** on an out-of-range fill-rate/demand combination (`SafetyStockRangeError`), naming the requested fill rate, the resolved mean/std, and which bound (±6σ) was hit. `on_out_of_range="clip"` opts back into today's silent-clip behavior explicitly.
- **`KRmseSafetyStock`/`KMaeSafetyStock` without `actuals`** raise `ValueError` at `ReplenishmentPolicy` construction, not deep inside a simulation run.
- **`calibration.optimize(...)` raises if the validation slice has fewer than 2 periods** — too small to mean anything, rather than silently reporting a "winning" factor fit to noise.

## 6. Testing

- **Per-strategy unit tests**: each `SafetyStockStrategy` / `OrderTrigger` tested in isolation against synthetic series — no `ReplenishmentPolicy` or simulator required. (Today, testing `k_rmse` alone requires instantiating one of the 20-field dataclasses; this is the direct payoff of the split.)
- **Parity tests**: re-run janrth's existing notebook scenarios (`mean_forecast_safety_stock_example`, `k_rmse_safety_stock_optimization_example`, `percentile_optimization_example`, `stock_replenishment_example`) through the new classes. Formulas that are unchanged (`sqrt_horizon`, `k_rmse`, `k_mae`, in-range `fill_rate`) must match old output to float tolerance. Any place output legitimately differs — `fill_rate` now raising instead of clipping; `calibration` picking a different factor because of the train/validation split — is asserted explicitly as an intentional behavior change, not silently accepted.
- **Golden simulation test**: port `stress_test_example.ipynb`'s scenario; assert aggregate simulated cost lands within tolerance of the original run's.
- **New-behavior tests**: (a) `FillRateSafetyStock` raises `SafetyStockRangeError` on a constructed out-of-range input; (b) `calibration.optimize` picks a different (better, out-of-sample) factor than the old in-sample-only method on synthetic data engineered so the two diverge.

## 7. Repo & Notebooks

New standalone repo, `daybreakai/replenishment`. Structure mirrors §2's layers under `src/replenishment/`. All 10 example notebooks under `notebooks/` are rewritten against the new API (no compat shim for the old long function names) — they double as integration tests / usage documentation for the new classes.

## 8. Explicitly Out of Scope

- No demand-shape classification (Syntetos-Boylan / King's formula / compound-Poisson) ported from Daybreak. Safety-stock strategies remain forecast-residual-based only.
- No multi-echelon (MEIO) support.
- No capacity/lot-sizing constraints on the resulting order quantity.
- No cross-SKU correlation handling.
