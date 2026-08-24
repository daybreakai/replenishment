import pandas as pd
from replenishment.io_ import (
    generate_standard_simulation_rows, standard_simulation_rows_to_dataframe,
    standard_simulation_rows_from_dataframe, split_standard_simulation_rows,
    build_policy_from_standard_rows,
)
from replenishment.strategies.safety_stock import SqrtHorizonSafetyStock
from replenishment.strategies.order_trigger import OrderUpToTrigger


def test_generate_standard_simulation_rows_produces_expected_row_count():
    rows = generate_standard_simulation_rows(
        n_unique_ids=3, periods=10, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    assert len(rows) == 3 * 10


def test_roundtrip_through_dataframe():
    rows = generate_standard_simulation_rows(
        n_unique_ids=2, periods=5, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    df = standard_simulation_rows_to_dataframe(rows, library="pandas")
    assert isinstance(df, pd.DataFrame)
    roundtripped = standard_simulation_rows_from_dataframe(df, cutoff=df["ds"].max())
    assert len(roundtripped) == len(rows)


def test_build_policy_from_standard_rows_returns_a_working_policy():
    rows = generate_standard_simulation_rows(
        n_unique_ids=1, periods=20, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    backtest_rows, _ = split_standard_simulation_rows(rows)
    policy = build_policy_from_standard_rows(
        backtest_rows, safety_stock_builder=lambda: SqrtHorizonSafetyStock(factor=1.65),
        trigger=OrderUpToTrigger(), lead_time=1,
    )
    assert policy.lead_time == 1
