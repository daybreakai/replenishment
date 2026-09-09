# Strategy selection by demand shape

Manual lookup table — nothing here is called automatically.
`replenishment.classification.classify_demand(history)` implements the same
Syntetos-Boylan ADI/CV² thresholds this table is built from, but has zero
callers anywhere in the codebase; use it by hand (`classify_demand(history)`)
to get one of the four labels below, then pick a strategy family from this
table and confirm its hyperparameters with `describe(name)`.

| Demand shape | ADI / CV² | Candidate strategies | Why |
|---|---|---|---|
| **Smooth** | ADI ≤ 1.32, CV² ≤ 0.49 | `KRmseSafetyStock`, `SqrtHorizonSafetyStock`, `FillRateSafetyStock` | Regular, low-variance demand — forecast-error-based buffers (RMSE/MAE-driven) work well; `FillRateSafetyStock` if you have a specific fill-rate target rather than just a k-sigma multiplier. |
| **Erratic** | ADI ≤ 1.32, CV² > 0.49 | `NegativeBinomialSafetyStock` | Regular timing but highly variable quantity — overdispersed count model fits better than a normal-error assumption. |
| **Intermittent** | ADI > 1.32, CV² ≤ 0.49 | `CompoundPoissonSafetyStock`, `NegativeBinomialSafetyStock` | Sparse demand, low variance when it occurs — the RMSE-based strategies above are unreliable with mostly-zero history. |
| **Lumpy** | ADI > 1.32, CV² > 0.49 | `CompoundPoissonSafetyStock`, `NegativeBinomialSafetyStock` | Sparse *and* highly variable when it occurs — the hardest case; both distributional strategies model this directly, try both and compare `fill_rate`/`total_cost` via the backtest script. |

Two strategies don't fit this table because they don't compute a buffer from
demand shape at all:

- `MultiplierSafetyStockStrategy` — flat `forecast * (multiplier - 1)`, useful
  when you already have a target multiplier from elsewhere (e.g. a client
  policy) rather than deriving one from history.
- `NullSafetyStockStrategy` — zero buffer, for forecasts that already embed
  their own buffer (e.g. a percentile forecast used directly as the
  order-up-to target).

`DemandBufferDecorator` wraps any of the above to add trend-chasing uplift
when the current forecast is running above a reference baseline — layer it on
top of whichever base strategy the table above suggests, it isn't a
replacement for one.

`KingsFormulaSafetyStock` sits closest to the "smooth" row (assumes
demand-lead_time independence, roughly normal demand) but additionally models
lead-time variance (`std_lead_time_periods`) — reach for it specifically when
lead time itself is uncertain, not just demand.
