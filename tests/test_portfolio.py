import pytest

from replenishment import Portfolio
from replenishment.io_ import (
    build_lead_time_forecast_article_configs_from_standard_rows,
    generate_standard_simulation_rows,
    standard_simulation_rows_to_dataframe,
)


def _rows(n=3, periods=12, lead_time=2, seed=7):
    return generate_standard_simulation_rows(
        n_unique_ids=n, periods=periods, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=lead_time,
        seed=seed,
    )


def test_simulate_matches_legacy_lead_time_builder_path():
    rows = _rows()
    lead_times = {r.unique_id: r.lead_time for r in rows}
    port = Portfolio(rows)
    res = port.simulate(factor=1.65, horizon=lead_times)
    with pytest.warns(DeprecationWarning):
        legacy_configs = build_lead_time_forecast_article_configs_from_standard_rows(
            rows, service_level_factor=1.65, safety_stock_method="sqrt_horizon",
            forecast_horizon=lead_times,
        )
    legacy = {uid: cfg.simulate() for uid, cfg in legacy_configs.items()}
    assert res.summaries.keys() == legacy.keys()
    for uid in legacy:
        assert res[uid].summary == legacy[uid].summary


def test_rop_mode_produces_different_orders_than_base_stock():
    rows = _rows()
    port = Portfolio(rows)
    base = port.simulate(factor=1.65, horizon=2)
    rop = port.simulate(factor=1.65, horizon=2, mode="rop")
    base_orders = sum(s.ordering_cost for s in base.summaries.values())
    rop_orders = sum(s.ordering_cost for s in rop.summaries.values())
    assert rop_orders < base_orders  # rop skips periods above the reorder point


def test_summary_frame_shape_and_columns():
    port = Portfolio(_rows(n=2))
    frame = port.simulate(factor=1.0).summary_frame()
    assert len(frame) == 2
    assert list(frame.columns) == [
        "fill_rate", "total_cost", "avg_on_hand",
        "holding_cost", "stockout_cost", "ordering_cost",
    ]


def test_portfolio_metrics_weighted_fill_and_worst_month():
    port = Portfolio(_rows())
    res = port.simulate(factor=1.65, horizon=2)
    metrics = res.portfolio_metrics()
    summaries = res.summaries.values()
    expected_weighted = sum(s.total_fulfilled for s in summaries) / sum(
        s.total_demand for s in summaries
    )
    assert metrics["weighted_fill"] == pytest.approx(expected_weighted)
    assert 0.0 <= metrics["worst_month_fill"] <= metrics["weighted_fill"] + 1e-9
    assert metrics["total_cost"] == pytest.approx(
        metrics["total_holding"] + metrics["total_stockout"] + metrics["total_ordering"]
    )


def test_from_dataframe_roundtrip_and_per_item_factor():
    rows = _rows(n=2)
    df = standard_simulation_rows_to_dataframe(rows, library="pandas")
    port = Portfolio.from_dataframe(df, cutoff=df["ds"].max())
    uids = port.unique_ids
    per_item = dict.fromkeys(uids, 1.65)
    assert port.simulate(factor=per_item).summaries.keys() == set(uids)


def test_empty_portfolio_rejected():
    with pytest.raises(ValueError):
        Portfolio([])
