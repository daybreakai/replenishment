"""Assemble the sbd_poc3 local fixture from SBD's raw flat files in ~/Downloads.

Produces fixtures/sbd_poc3/sbd_poc3.parquet in this repo's own column shape
(unique_id, ds, demand, forecast, lead_time, holding_cost_per_unit,
stockout_cost_per_unit, order_cost_per_order, current_stock) — see
fixtures/sbd_poc3/README.md.

Join keys and unit-normalization follow Sebastian Urbina's sbd-poc pipeline
(github.com/seburbina/sbd-poc, fva-pipeline branch) exactly, since that's the
already-vetted, already-running logic for this data:
  - pid = PRODUCT_MATERIAL_CODE (Profitability) — matched DIRECTLY against
    MFG_Lead_Time.ITEM and "Current Inventory Values by Material".PROD_KEY.
    No PROD_ID_PM_ORG crosswalk (that was this repo's own earlier, more
    conservative approach on the supplyplanning side — Sebastian's code
    doesn't bridge, so this doesn't either, to stay consistent with the one
    pipeline that's actually produced validated SBD results).
  - Lead time: MINUTES/HOURS/DAYS/WEEKS -> weeks, median across locations,
    default 8 wks when missing (seg_02_forecast.py). Converted to months
    (/4.345) for lead_time here since ds grain is monthly.
  - unit_cost = trailing-period (FY*100+FM >= 202502) COS_TOTAL / NET_SHIP_QTY
    per material (seg_03c_margin_local.py's true-COGS method).
  - forecast = same-calendar-month demand one year prior — Sebastian's own
    naive-forecast definition used as the FVA/model-lift benchmark
    (seg_05_assemble.py's model_lift()). Not JDA's forecast (that's
    GLOBAL_FORECAST_SNAPSHOT, keyed on PROD_KEY/LOC/CUST — a different grain
    entirely, never merged onto material_code anywhere in his pipeline).

Known gaps, not hidden:
  - current_stock is a single on-hand snapshot (no on-hand time series exists
    in what's been shared), so it's the SAME value repeated across every
    (unique_id, ds) row for that unique_id — not a true per-period actual.
  - holding_cost_per_unit = unit_cost * 25%/yr / 12. 25%/yr is a standard
    textbook holding-cost-rate default, not an SBD-specific number — nobody
    has given us SBD's actual carrying cost rate.
  - stockout_cost_per_unit = (ASP - unit_cost), i.e. lost margin per unit —
    the standard proxy absent an actual expedite-cost figure.
  - order_cost_per_order = 12.5, the replenishment package's own built-in
    default (io_.py). No SBD ordering-cost data exists anywhere in what's
    been shared; this is a placeholder, not a fitted number.
  - First 12 months of each material's history have no forecast (no prior
    year to look back to) and are dropped.

Usage:
    python3 scripts/assemble_sbd_fixture.py
    python3 scripts/assemble_sbd_fixture.py --downloads ~/Downloads --out fixtures/sbd_poc3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

HOLDING_RATE_ANNUAL = 0.25
ORDER_COST_DEFAULT = 12.5
DEFAULT_LEAD_WKS = 8.0


def _demand_panel(downloads: Path) -> pl.LazyFrame:
    prof = pl.scan_csv(
        downloads / "Profitability_20230501-20260430.csv",
        schema_overrides={"NET_SHIP_QTY": pl.Float64, "NET_SALES": pl.Float64, "COS_TOTAL": pl.Float64},
    )
    return (
        prof.filter(pl.col("FISCAL_MONTH").is_between(1, 12))
        .group_by(["PRODUCT_MATERIAL_CODE", "FISCAL_YEAR", "FISCAL_MONTH"])
        .agg(
            pl.col("NET_SHIP_QTY").sum().alias("qty"),
            pl.col("NET_SALES").sum().alias("net_sales"),
            pl.col("COS_TOTAL").sum().alias("cos_total"),
        )
        .rename({"PRODUCT_MATERIAL_CODE": "pid"})
    )


def _lead_time_months(downloads: Path) -> pl.LazyFrame:
    lt = pl.scan_csv(downloads / "MFG_Lead_Time.csv", infer_schema_length=10000)
    wks = (
        pl.when(pl.col("UNITS") == "MINUTES").then(pl.col("LOCK_OR_LEADTIME") / 10080.0)
        .when(pl.col("UNITS") == "HOURS").then(pl.col("LOCK_OR_LEADTIME") / 168.0)
        .when(pl.col("UNITS") == "DAYS").then(pl.col("LOCK_OR_LEADTIME") / 7.0)
        .when(pl.col("UNITS") == "WEEKS").then(pl.col("LOCK_OR_LEADTIME"))
        .otherwise(None)
        .alias("lead_wks")
    )
    return (
        lt.with_columns(wks)
        .group_by("ITEM")
        .agg(pl.col("lead_wks").median())
        .rename({"ITEM": "pid"})
    )


def _on_hand(downloads: Path) -> pl.LazyFrame:
    return pl.scan_csv(downloads / "Current Inventory Values by Material.csv").select(
        pl.col("PROD_KEY").alias("pid"),
        pl.col("TOTAL_ON_HAND_QTY").alias("current_stock"),
    )


def assemble(downloads: Path) -> pl.DataFrame:
    demand = _demand_panel(downloads).collect(engine="streaming")
    lead = _lead_time_months(downloads).collect(engine="streaming")
    onhand = _on_hand(downloads).collect(engine="streaming")

    # trailing-period unit cost + ASP per material (Sebastian's true-COGS method)
    cost = (
        demand.filter((pl.col("FISCAL_YEAR") * 100 + pl.col("FISCAL_MONTH")) >= 202502)
        .group_by("pid")
        .agg(
            pl.col("qty").sum().alias("t_qty"),
            pl.col("net_sales").sum().alias("t_ns"),
            pl.col("cos_total").sum().alias("t_cos"),
        )
        .with_columns(
            (pl.col("t_ns") / pl.col("t_qty")).alias("asp"),
            (pl.col("t_cos") / pl.col("t_qty")).alias("unit_cost"),
        )
        .select("pid", "asp", "unit_cost")
    )
    median_unit_cost = cost["unit_cost"].median()

    demand = demand.with_columns(
        pl.date(pl.col("FISCAL_YEAR"), pl.col("FISCAL_MONTH"), 1).alias("ds")
    ).sort(["pid", "ds"])

    # naive seasonal forecast = same calendar month, one year prior (per-pid, ordered by ds)
    demand = demand.with_columns(pl.col("qty").shift(12).over("pid").alias("forecast"))

    out = (
        demand.join(lead, on="pid", how="left")
        .join(onhand, on="pid", how="left")
        .join(cost, on="pid", how="left")
        .filter(pl.col("forecast").is_not_null())
        .with_columns(
            pl.col("lead_wks").fill_null(DEFAULT_LEAD_WKS),
            pl.col("current_stock").fill_null(0.0),
            pl.col("unit_cost").fill_null(median_unit_cost),
        )
        .with_columns(
            (pl.col("lead_wks") / 4.345).round(0).clip(lower_bound=0).cast(pl.Int64).alias("lead_time"),
            (pl.col("unit_cost") * HOLDING_RATE_ANNUAL / 12.0).alias("holding_cost_per_unit"),
            (pl.col("asp") - pl.col("unit_cost")).clip(lower_bound=0).fill_null(0.0).alias("stockout_cost_per_unit"),
            pl.lit(ORDER_COST_DEFAULT).alias("order_cost_per_order"),
            pl.col("pid").alias("unique_id"),
            pl.col("qty").round(0).cast(pl.Int64).alias("demand"),
            pl.col("forecast").round(0).cast(pl.Int64),
            pl.col("current_stock").round(0).cast(pl.Int64),
        )
        .select(
            "unique_id", "ds", "demand", "forecast", "lead_time",
            "holding_cost_per_unit", "stockout_cost_per_unit", "order_cost_per_order",
            "current_stock",
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", default="~/Downloads", type=str)
    parser.add_argument("--out", default="fixtures/sbd_poc3", type=str)
    args = parser.parse_args()

    downloads = Path(args.downloads).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = assemble(downloads)
    out_path = out_dir / "sbd_poc3.parquet"
    df.write_parquet(out_path)

    print(f"wrote {len(df):,} rows ({df['unique_id'].n_unique():,} materials) -> {out_path}")
    print(f"ds range: {df['ds'].min()} .. {df['ds'].max()}")
    print(f"current_stock: {(df['current_stock'] > 0).sum():,} materials with nonzero on-hand")
    print(f"unit_cost imputed (median fill) for materials outside trailing-period window — see docstring")


if __name__ == "__main__":
    main()
