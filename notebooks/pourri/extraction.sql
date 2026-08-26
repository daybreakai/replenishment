-- Pourri monthly forecast-vs-actuals extraction (pourri_prod_sc).
-- Produces notebooks/pourri/pourri_monthly.csv at VINTAGE grain: one row per
-- product x target month x snapshot_date (all products with full 10/10 month
-- actuals coverage; 116 products, 3675 rows as of 2026-08-25). Freshest
-- vintage per (product, month) = min timestep — the notebook dedupes to that
-- for the sim; the full grain enables performance-by-snapshot analysis.
--
-- Construction notes:
-- * Forecasts: model_store.t_best_model_forecast, plan_type='ensembler'.
--   Snapshots carry up to 42 duplicate plan runs -> dedupe to latest
--   job_run_id per (snapshot, product, ship_to, timestep).
-- * forecast_start_dttm convention shifts mid-2026 (timestep 1 = next month
--   early, = snapshot month later) -> derive target month as
--   add_months(month(forecast_start_dttm), timestep - 1). Within a snapshot,
--   one timestep per target month (min timestep kept defensively).
-- * Actuals: data_store.t_outbound_shipment, monthly sum of shipped_qty per
--   product (aggregated across sites), window 2025-10-01 .. 2026-07-31
--   (2026-08 excluded: partial month).
-- * unit_cost: real, from t_outbound_shipment. price: real, avg
--   customer_effective_price supplementary series.
-- * All products with full 10/10 month actuals coverage (no top-N cut).

WITH deduped AS (
  SELECT product_id, ship_to_site_id, snapshot_date, timestep, forecast_start_dttm, mean
  FROM pourri_prod_sc.model_store.t_best_model_forecast
  WHERE plan_type='ensembler' AND snapshot_date >= '2025-09-01'
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY snapshot_date, product_id, ship_to_site_id, timestep
    ORDER BY job_run_id DESC, load_timestamp DESC) = 1
),
fc_by_month AS (
  SELECT product_id, snapshot_date,
         add_months(date_trunc('month', forecast_start_dttm), CAST(timestep AS INT) - 1) AS month,
         timestep,
         SUM(mean) AS fc
  FROM deduped
  GROUP BY 1, 2, 3, 4
),
vintages AS (
  SELECT product_id, snapshot_date, month, CAST(timestep AS INT) AS timestep, fc
  FROM fc_by_month
  QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id, snapshot_date, month ORDER BY timestep ASC) = 1
),
actuals AS (
  SELECT product_id, date_trunc('month', actual_ship_date) AS month,
         SUM(shipped_qty) AS qty, AVG(unit_cost) AS unit_cost
  FROM pourri_prod_sc.data_store.t_outbound_shipment
  WHERE actual_ship_date >= '2025-10-01' AND actual_ship_date < '2026-08-01'
  GROUP BY 1, 2
),
price AS (
  SELECT product_id, AVG(time_series_value) AS avg_price
  FROM pourri_prod_sc.data_store.t_mapped_supplementary_time_series_customer_effective_price
  WHERE time_series_value > 0
  GROUP BY 1
),
joined AS (
  SELECT a.product_id, v.snapshot_date, v.timestep, a.month, a.qty, v.fc, a.unit_cost, p.avg_price
  FROM actuals a
  JOIN vintages v ON a.product_id = v.product_id AND a.month = v.month
  LEFT JOIN price p ON a.product_id = p.product_id
),
complete_products AS (
  SELECT product_id
  FROM joined GROUP BY 1 HAVING COUNT(DISTINCT month) = 10
)
SELECT j.product_id, DATE(j.snapshot_date) AS snapshot_date, j.timestep,
       DATE(j.month) AS month, ROUND(j.qty) AS actuals,
       ROUND(j.fc, 1) AS forecast, ROUND(j.unit_cost, 3) AS unit_cost,
       ROUND(j.avg_price, 3) AS price
FROM joined j JOIN complete_products c ON j.product_id = c.product_id
ORDER BY j.product_id, j.month, j.snapshot_date
