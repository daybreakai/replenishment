import csv
import tempfile
from pathlib import Path

import pandas as pd
from replenishment.io_ import (
    generate_standard_simulation_rows, standard_simulation_rows_to_dataframe,
    standard_simulation_rows_from_dataframe, split_standard_simulation_rows,
    build_policy_from_standard_rows, load_standard_simulation_rows,
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


_BASE_FIELDNAMES = [
    "unique_id", "ds", "demand", "forecast", "actuals", "holding_cost_per_unit",
    "stockout_cost_per_unit", "order_cost_per_order", "lead_time", "initial_on_hand",
    "current_stock",
]


def _write_csv(path: Path, actuals_col: str = "actuals", lead_time_col: str = "lead_time") -> None:
    fieldnames = [f if f not in ("actuals", "lead_time") else
                  {"actuals": actuals_col, "lead_time": lead_time_col}[f]
                  for f in _BASE_FIELDNAMES]
    rows = [{
        "unique_id": "A", "ds": "2024-01-01", "demand": 10, "forecast": 10,
        actuals_col: 10, "holding_cost_per_unit": 1.0, "stockout_cost_per_unit": 5.0,
        "order_cost_per_order": 0.0, lead_time_col: 3, "initial_on_hand": 20, "current_stock": 20,
    }]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_standard_simulation_rows_reads_csv_with_default_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.csv"
        _write_csv(path)
        rows = load_standard_simulation_rows(str(path))
        assert len(rows) == 1
        assert rows[0].actuals == 10
        assert rows[0].lead_time == 3


def test_load_standard_simulation_rows_honors_actuals_field_override():
    # This is the exact drift this function exists to prevent: a renamed
    # actuals column must resolve identically for every caller that passes
    # actuals_field, not just the one that happened to test it first.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.csv"
        _write_csv(path, actuals_col="actual_units")
        rows = load_standard_simulation_rows(str(path), actuals_field="actual_units")
        assert rows[0].actuals == 10


def test_load_standard_simulation_rows_honors_lead_time_field_override():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.csv"
        _write_csv(path, lead_time_col="supplier_lead_time")
        rows = load_standard_simulation_rows(str(path), lead_time_field="supplier_lead_time")
        assert rows[0].lead_time == 3


def test_load_standard_simulation_rows_rejects_unsupported_extension():
    try:
        load_standard_simulation_rows("nope.txt")
    except ValueError as exc:
        assert "nope.txt" in str(exc)
        return
    raise AssertionError("expected ValueError for an unsupported extension")


def test_article_config_builders_accept_per_item_moq():
    from replenishment.io_ import (
        build_lead_time_forecast_article_configs_from_standard_rows,
        build_point_forecast_article_configs_from_standard_rows,
    )
    rows = generate_standard_simulation_rows(
        n_unique_ids=2, periods=20, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    moqs = {"A": 30, "B": 12}
    point = build_point_forecast_article_configs_from_standard_rows(
        rows, service_level_factor=1.65, safety_stock_method="sqrt_horizon", moq=moqs,
    )
    import pytest
    with pytest.warns(DeprecationWarning):
        lead = build_lead_time_forecast_article_configs_from_standard_rows(
            rows, service_level_factor=1.65, safety_stock_method="sqrt_horizon", moq=moqs,
        )
    assert point["A"].policy.moq == 30 and point["B"].policy.moq == 12
    assert lead["A"].policy.moq == 30 and lead["B"].policy.moq == 12


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


def test_point_builder_honors_per_item_fixed_error():
    from replenishment.io_ import build_point_forecast_article_configs_from_standard_rows
    from replenishment.strategies.safety_stock import FixedErrorSafetyStock, KRmseSafetyStock
    rows = generate_standard_simulation_rows(
        n_unique_ids=2, periods=20, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    configs = build_point_forecast_article_configs_from_standard_rows(
        rows, service_level_factor=2.0, safety_stock_method="k_rmse",
        fixed_error={"A": 3.0, "B": 5.0},
    )
    ss_a = configs["A"].policy.safety_stock
    assert isinstance(ss_a, FixedErrorSafetyStock)
    assert ss_a.error == 3.0 and ss_a.factor == 2.0 and not ss_a.scale_by_horizon
    assert configs["B"].policy.safety_stock.error == 5.0
    # without fixed_error the method's history-based strategy is unchanged
    plain = build_point_forecast_article_configs_from_standard_rows(
        rows, service_level_factor=2.0, safety_stock_method="k_rmse",
    )
    assert isinstance(plain["A"].policy.safety_stock, KRmseSafetyStock)


def test_fixed_error_sqrt_horizon_method_selects_scaled_shape():
    from replenishment.io_ import build_point_forecast_article_configs_from_standard_rows
    rows = generate_standard_simulation_rows(
        n_unique_ids=1, periods=10, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    configs = build_point_forecast_article_configs_from_standard_rows(
        rows, service_level_factor=1.65, safety_stock_method="sqrt_horizon",
        fixed_error=4.0,
    )
    assert configs["A"].policy.safety_stock.scale_by_horizon


def test_fill_rate_method_maps_factor_to_target_fill_rate():
    from replenishment.io_ import build_point_forecast_article_configs_from_standard_rows
    from replenishment.strategies.safety_stock import FillRateSafetyStock
    rows = generate_standard_simulation_rows(
        n_unique_ids=1, periods=20, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    configs = build_point_forecast_article_configs_from_standard_rows(
        rows, service_level_factor=0.95, safety_stock_method="fill_rate",
    )
    ss = configs["A"].policy.safety_stock
    assert isinstance(ss, FillRateSafetyStock)
    assert ss.target_fill_rate == 0.95


def test_compound_poisson_method_maps_factor_to_target_service_level():
    from replenishment.io_ import build_point_forecast_article_configs_from_standard_rows
    from replenishment.strategies.distributional_safety_stock import CompoundPoissonSafetyStock
    rows = generate_standard_simulation_rows(
        n_unique_ids=1, periods=20, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    configs = build_point_forecast_article_configs_from_standard_rows(
        rows, service_level_factor=0.90, safety_stock_method="compound_poisson",
    )
    ss = configs["A"].policy.safety_stock
    assert isinstance(ss, CompoundPoissonSafetyStock)
    assert ss.target_service_level == 0.90


def test_negative_binomial_method_maps_factor_to_target_service_level():
    from replenishment.io_ import build_point_forecast_article_configs_from_standard_rows
    from replenishment.strategies.distributional_safety_stock import NegativeBinomialSafetyStock
    rows = generate_standard_simulation_rows(
        n_unique_ids=1, periods=20, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    configs = build_point_forecast_article_configs_from_standard_rows(
        rows, service_level_factor=0.90, safety_stock_method="negative_binomial",
    )
    ss = configs["A"].policy.safety_stock
    assert isinstance(ss, NegativeBinomialSafetyStock)
    assert ss.target_service_level == 0.90
    # builder-level default: a portfolio-wide sweep shouldn't abort on one
    # non-overdispersed item (see _safety_stock_builder_for_method)
    assert ss.on_underdispersion == "zero"


def test_fixed_error_rejected_for_probability_parameterized_methods():
    import pytest
    from replenishment.io_ import build_point_forecast_article_configs_from_standard_rows
    rows = generate_standard_simulation_rows(
        n_unique_ids=1, periods=20, history_mean=10, history_std=2,
        forecast_mean=10, forecast_std=1, holding_cost_per_unit=1,
        stockout_cost_per_unit=5, order_cost_per_order=2, lead_time=1, seed=7,
    )
    with pytest.raises(ValueError, match="no forecast-error series"):
        build_point_forecast_article_configs_from_standard_rows(
            rows, service_level_factor=0.95, safety_stock_method="fill_rate",
            fixed_error=4.0,
        )
