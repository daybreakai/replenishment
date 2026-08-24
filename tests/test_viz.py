import matplotlib
matplotlib.use("Agg")
import pandas as pd
from replenishment.viz import plot_replenishment_decisions


def test_plot_replenishment_decisions_runs_without_error():
    # ponytail: brief's fixture used column names ("demand"/"order_quantity")
    # that don't match io_.py's actual StandardSimulationRow/
    # ReplenishmentDecisionRow dataframe schema (which uses "actuals" and
    # "quantity" respectively -- verified against
    # standard_simulation_rows_to_dicts / replenishment_decision_rows_to_dicts
    # in src/replenishment/io_.py). Adjusted column names to match; same
    # values and shape as the brief's smoke test.
    raw_df = pd.DataFrame({
        "unique_id": ["a"] * 5, "ds": pd.date_range("2024-01-01", periods=5),
        "actuals": [10, 12, 8, 11, 9], "forecast": [10, 10, 10, 10, 10],
        "current_stock": [20, 15, 20, 15, 12],
    })
    decision_df = pd.DataFrame({
        "unique_id": ["a"] * 5, "ds": pd.date_range("2024-01-01", periods=5),
        "quantity": [0, 5, 0, 8, 0], "safety_stock": [5, 5, 5, 5, 5],
    })
    fig_or_none = plot_replenishment_decisions(raw_df, decision_df, unique_id="a")
    assert fig_or_none is None or fig_or_none is not None  # smoke test: no exception raised
