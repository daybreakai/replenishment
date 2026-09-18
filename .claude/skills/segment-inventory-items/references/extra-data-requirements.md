# Extra data requirements by strategy

Every segmentation run always pulls `t_outbound_shipment` (the base demand
signal) and `t_product` (for `unit_cost`) -- see `SEGMENTATION_STRATEGIES` in
`../scripts/segmentation_strategies.py`, the source of truth
`validate_segmentation.py` checks against. This file explains *why* each
strategy needs what it needs; it doesn't duplicate the table structure
itself.

| Strategy | Extra tables beyond the default pull | Why |
|---|---|---|
| `adi_cv2` | none | Pure demand-timing/variability (ADI/CV²) -- doesn't touch price at all. |
| `abc_revenue` | none | Revenue = `shipped_qty * t_product.unit_cost`, already in the default pull. |
| `xyz_variability` | none | Same inputs as `abc_revenue`, just a CV of the per-period revenue series instead of a Pareto rank. |
| `abc_xyz_matrix` | none | Cross of the two above; no new data, just combines outputs already computed. |
| `abc_revenue_by_hierarchy` | `t_product_hierarchy` | Rolls revenue up to `t_product.product_group_id` (via `t_product_hierarchy.parent_product_group_id`) *before* bucketing, so a strategy that's supposed to segment a product group can't accidentally run on a flat per-SKU cut with no hierarchy join. |

## Documented, not implemented

**Promo-adjusted segmentation** -- distinguishing organic demand variability
from promo-driven spikes (e.g. by comparing `t_supplementary_time_series`
`item_rate` vs. `customer_effective_price` to detect a discount period) is a
real idea, but this client schema (`pourri_prod_sc.data_store`, checked
2026-09-18) has no dedicated promos table, and inferring "promo" from a
price-series divergence is a business-rule guess, not a documented fact.
Don't build this without confirming the signal with the user first --
propose it as a new `SEGMENTATION_STRATEGIES` entry requiring
`t_supplementary_time_series`, not a silent addition to an existing one.
