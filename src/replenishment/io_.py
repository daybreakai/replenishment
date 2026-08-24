"""Helpers for loading bulk optimization inputs.

Ported from janrth's io.py. Every function that used to construct one of
the 7 duplicated old policy classes (PointForecastOptimizationPolicy,
RopPointForecastOptimizationPolicy, LeadTimeForecastOptimizationPolicy, and
their percentile-forecast siblings) now builds the single unified
ReplenishmentPolicy (Task 8), via one shared helper --
build_policy_from_standard_rows -- or a direct
ReplenishmentPolicy.order_up_to/.reorder_point call.

Scope note: janrth's optimization.py/aggregation.py/service_levels.py
ecosystem (ArticleSimulationConfig, ForecastCandidatesConfig,
simulate_replenishment_for_articles, optimize_service_level_factors, and
the decision-row builders keyed off janrth's old *OptimizationResult
types) was never ported into this repo across Tasks 1-10, and is not part
of this task's produces list. Where the functions below used to hand off
into that ecosystem, this module defines the minimal local stand-ins
needed to keep them self-contained (ArticleSimulationConfig,
ForecastCandidatesConfig below) rather than importing modules that don't
exist here. `build_replenishment_decisions_from_simulations`,
`build_replenishment_decisions_from_optimization_results`, and
`_decision_metadata_from_result` are not ported for the same reason: they
hard-depend on janrth's AggregationServiceLevelOptimizationResult /
AggregationForecastTargetOptimizationResult / etc. and on a
SimulationResult.metadata field this repo's SimulationResult doesn't have.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
import random
import string
import warnings

from replenishment.calibration import CalibrationResult, optimize as calibration_optimize
from replenishment.policy import ReplenishmentPolicy
from replenishment.simulation import SimulationResult, simulate_replenishment
from replenishment.strategies.multiplier import NullSafetyStockStrategy
from replenishment.strategies.order_trigger import OrderUpToTrigger, ReorderPointTrigger
from replenishment.strategies.safety_stock import KMaeSafetyStock, KRmseSafetyStock, SqrtHorizonSafetyStock
from replenishment.timeseries import TimeSeries


@dataclass(frozen=True)
class PointForecastRow:
    unique_id: str
    period: int
    demand: int
    forecast: int
    actual: int


@dataclass(frozen=True)
class PercentileForecastRow:
    unique_id: str
    period: int
    demand: int
    target: float | str
    forecast: int


@dataclass(frozen=True)
class StandardSimulationRow:
    unique_id: str
    ds: str
    demand: int
    forecast: int
    actuals: int | float | None
    holding_cost_per_unit: float
    stockout_cost_per_unit: float
    order_cost_per_order: float
    lead_time: int
    initial_on_hand: int
    current_stock: int
    forecast_percentiles: Mapping[str, int]
    is_forecast: bool = False


@dataclass(frozen=True)
class ReplenishmentDecisionRow:
    unique_id: str
    ds: str
    quantity: int
    demand: int | None = None
    forecast_quantity: float | None = None
    forecast_quantity_lead_time: float | None = None
    reorder_point: float | None = None
    order_up_to: float | None = None
    incoming_stock: int | None = None
    starting_stock: int | None = None
    ending_stock: int | None = None
    safety_stock: float | None = None
    starting_on_hand: int | None = None
    ending_on_hand: int | None = None
    current_stock: int | None = None
    on_order: int | None = None
    backorders: int | None = None
    missed_sales: int | None = None
    sigma: float | None = None
    service_level_mode: str | None = None
    aggregation_window: int | None = None
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    percentile_target: float | str | None = None


@dataclass(frozen=True)
class ReplenishmentDecisionMetadata:
    sigma: float | None = None
    service_level_mode: str | None = None
    aggregation_window: int | None = None
    review_period: int | None = None
    forecast_horizon: int | None = None
    rmse_window: int | None = None
    percentile_target: float | str | None = None


@dataclass(frozen=True)
class ArticleSimulationConfig:
    """Minimal per-article simulation config: everything simulate_replenishment
    needs for one unique_id, with `policy` now a single ReplenishmentPolicy
    instead of one of janrth's 7 duplicated policy classes. Replaces
    janrth's ArticleSimulationConfig (simulation.py), which this repo has
    not ported -- see module docstring."""

    periods: int
    demand: list[int]
    initial_on_hand: int
    lead_time: int
    policy: ReplenishmentPolicy
    holding_cost_per_unit: float = 0.0
    stockout_cost_per_unit: float = 0.0
    order_cost_per_order: float = 0.0
    order_cost_per_unit: float = 0.0

    def simulate(self) -> SimulationResult:
        return simulate_replenishment(
            periods=self.periods, demand=self.demand, initial_on_hand=self.initial_on_hand,
            lead_time=self.lead_time, policy=self.policy,
            holding_cost_per_unit=self.holding_cost_per_unit,
            stockout_cost_per_unit=self.stockout_cost_per_unit,
            order_cost_per_order=self.order_cost_per_order,
            order_cost_per_unit=self.order_cost_per_unit,
        )


@dataclass(frozen=True)
class ForecastCandidatesConfig:
    """Minimal per-article config for percentile-forecast-candidate
    evaluation: one ReplenishmentPolicy per candidate target (keyed the
    same way janrth's forecast_candidates dict was), each using
    NullSafetyStockStrategy since percentile forecasts already carry their
    own buffer. Replaces janrth's ForecastCandidatesConfig -- see module
    docstring."""

    periods: int
    demand: list[int]
    initial_on_hand: int
    lead_time: int
    review_period: int
    forecast_horizon: int
    candidate_policies: dict[float | str, ReplenishmentPolicy]
    holding_cost_per_unit: float = 0.0
    stockout_cost_per_unit: float = 0.0
    order_cost_per_order: float = 0.0
    order_cost_per_unit: float = 0.0

    def simulate(self, target: float | str) -> SimulationResult:
        return simulate_replenishment(
            periods=self.periods, demand=self.demand, initial_on_hand=self.initial_on_hand,
            lead_time=self.lead_time, policy=self.candidate_policies[target],
            holding_cost_per_unit=self.holding_cost_per_unit,
            stockout_cost_per_unit=self.stockout_cost_per_unit,
            order_cost_per_order=self.order_cost_per_order,
            order_cost_per_unit=self.order_cost_per_unit,
        )


_SAFETY_STOCK_STRATEGIES = {
    "sqrt_horizon": SqrtHorizonSafetyStock,
    "k_rmse": KRmseSafetyStock,
    "k_mae": KMaeSafetyStock,
}


def _safety_stock_builder_for_method(method: str | None, factor: float):
    """Map janrth's safety_stock_method string onto the new strategy
    classes. Only the three canonical method names are supported (per
    Task 11's brief) -- janrth's extra aliases ("legacy", "raw_rmse",
    "k*mae", ...) are not re-created here."""
    normalized = (method or "sqrt_horizon").strip().lower()
    strategy_cls = _SAFETY_STOCK_STRATEGIES.get(normalized)
    if strategy_cls is None:
        raise ValueError(
            f"safety_stock_method must be one of {sorted(_SAFETY_STOCK_STRATEGIES)}, got {method!r}."
        )
    return lambda: strategy_cls(factor=factor)


def build_policy_from_standard_rows(
    rows, *, safety_stock_builder, trigger, lead_time: int,
    review_period: int = 1, forecast_horizon: int = 1,
) -> ReplenishmentPolicy:
    """Shared ReplenishmentPolicy construction for a straightforward
    standard-row build with untrimmed, matching-length forecast/actuals.

    Not currently called by the builders below: the two eligible
    *_from_standard_rows functions handle trimmed/overridden actuals series
    that don't fit this helper's simple sort-and-zip contract (see
    task-11-report.md). Provided as reusable infrastructure for callers with
    plain matching-length rows -- exercised directly by test_io_.py."""
    forecast_values = [row.forecast for row in sorted(rows, key=lambda r: r.ds)]
    actual_values = [row.actuals for row in sorted(rows, key=lambda r: r.ds)]
    return ReplenishmentPolicy(
        forecast=TimeSeries.from_values(forecast_values),
        actuals=TimeSeries.from_values(actual_values),
        safety_stock=safety_stock_builder(),
        trigger=trigger, lead_time=lead_time,
        review_period=review_period, forecast_horizon=forecast_horizon,
    )


def generate_standard_simulation_rows(
    *,
    n_unique_ids: int,
    periods: int,
    start_date: str | date | datetime = "2024-01-01",
    frequency_days: int = 30,
    forecast_start_period: int | None = None,
    history_mean: float = 20.0,
    history_std: float = 5.0,
    forecast_mean: float = 20.0,
    forecast_std: float = 4.0,
    percentile_multipliers: Mapping[str, float] | None = None,
    holding_cost_per_unit: float = 0.5,
    stockout_cost_per_unit: float = 3.0,
    order_cost_per_order: float = 12.5,
    lead_time: int = 1,
    initial_on_hand: int = 30,
    current_stock: int | None = None,
    seed: int | None = None,
) -> list[StandardSimulationRow]:
    """Generate synthetic rows that match the standard simulation schema."""
    if n_unique_ids <= 0:
        raise ValueError("n_unique_ids must be positive.")
    if periods <= 0:
        raise ValueError("periods must be positive.")
    if frequency_days <= 0:
        raise ValueError("frequency_days must be positive.")
    if forecast_start_period is not None and not (0 <= forecast_start_period <= periods):
        raise ValueError("forecast_start_period must be within the period range.")

    if isinstance(start_date, str):
        base_date = date.fromisoformat(start_date)
    elif isinstance(start_date, datetime):
        base_date = start_date.date()
    else:
        base_date = start_date

    rng = random.Random(seed)

    def sample_int(mean: float, std: float) -> int:
        value = rng.gauss(mean, std)
        return max(0, int(round(value)))

    if percentile_multipliers is None:
        percentile_multipliers = {"p50": 1.0, "p90": 1.25}

    unique_ids: list[str] = []
    for index in range(n_unique_ids):
        if index < len(string.ascii_uppercase):
            unique_ids.append(string.ascii_uppercase[index])
        else:
            unique_ids.append(f"SKU-{index + 1:03d}")

    rows: list[StandardSimulationRow] = []
    for unique_id in unique_ids:
        for period in range(periods):
            is_forecast = (
                forecast_start_period is not None and period >= forecast_start_period
            )
            ds = (base_date + timedelta(days=period * frequency_days)).isoformat()
            forecast = sample_int(forecast_mean, forecast_std)
            actuals = None if is_forecast else sample_int(history_mean, history_std)
            demand = forecast if is_forecast else int(actuals)
            forecast_percentiles = {
                label: max(0, int(round(forecast * multiplier)))
                for label, multiplier in percentile_multipliers.items()
            }
            rows.append(
                StandardSimulationRow(
                    unique_id=unique_id,
                    ds=ds,
                    demand=demand,
                    forecast=forecast,
                    actuals=actuals,
                    holding_cost_per_unit=holding_cost_per_unit,
                    stockout_cost_per_unit=stockout_cost_per_unit,
                    order_cost_per_order=order_cost_per_order,
                    lead_time=lead_time,
                    initial_on_hand=initial_on_hand,
                    current_stock=initial_on_hand if current_stock is None else current_stock,
                    forecast_percentiles=forecast_percentiles,
                    is_forecast=is_forecast,
                )
            )
    return rows


def standard_simulation_rows_to_dicts(
    rows: Iterable[StandardSimulationRow],
    *,
    forecast_prefix: str = "forecast_",
) -> list[dict[str, str | int | float | bool]]:
    """Convert standard rows into dictionaries suitable for DataFrame or CSV use."""
    serialized: list[dict[str, str | int | float]] = []
    for row in rows:
        entry: dict[str, str | int | float | bool | None] = {
            "unique_id": row.unique_id,
            "ds": row.ds,
            "demand": row.demand,
            "forecast": row.forecast,
            "actuals": row.actuals,
            "holding_cost_per_unit": row.holding_cost_per_unit,
            "stockout_cost_per_unit": row.stockout_cost_per_unit,
            "order_cost_per_order": row.order_cost_per_order,
            "lead_time": row.lead_time,
            "initial_on_hand": row.initial_on_hand,
            "current_stock": row.current_stock,
            "is_forecast": row.is_forecast,
        }
        for label, value in row.forecast_percentiles.items():
            entry[f"{forecast_prefix}{label}"] = value
        serialized.append(entry)
    return serialized


def standard_simulation_rows_to_dataframe(
    rows: Iterable[StandardSimulationRow],
    *,
    library: str = "pandas",
    forecast_prefix: str = "forecast_",
    include_demand: bool = False,
):
    """Convert standard rows into a pandas or polars DataFrame."""
    data = standard_simulation_rows_to_dicts(rows, forecast_prefix=forecast_prefix)
    if not include_demand:
        for entry in data:
            entry.pop("demand", None)
    if library == "pandas":
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required for standard_simulation_rows_to_dataframe(library='pandas')."
            ) from exc
        return pd.DataFrame(data)
    if library == "polars":
        try:
            import polars as pl  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "polars is required for standard_simulation_rows_to_dataframe(library='polars')."
            ) from exc
        return pl.DataFrame(data)
    raise ValueError("library must be 'pandas' or 'polars'.")


def replenishment_decision_rows_to_dicts(
    rows: Iterable[ReplenishmentDecisionRow],
) -> list[dict[str, str | int | float | None]]:
    return [
        {
            "unique_id": row.unique_id,
            "ds": row.ds,
            "quantity": row.quantity,
            "demand": row.demand,
            "forecast_quantity": row.forecast_quantity,
            "forecast_quantity_lead_time": row.forecast_quantity_lead_time,
            "reorder_point": row.reorder_point,
            "order_up_to": row.order_up_to,
            "incoming_stock": row.incoming_stock,
            "starting_stock": row.starting_stock,
            "ending_stock": row.ending_stock,
            "safety_stock": row.safety_stock,
            "starting_on_hand": row.starting_on_hand,
            "ending_on_hand": row.ending_on_hand,
            "current_stock": row.current_stock,
            "on_order": row.on_order,
            "backorders": row.backorders,
            "missed_sales": row.missed_sales,
            "sigma": row.sigma,
            "aggregation_window": row.aggregation_window,
            "review_period": row.review_period,
            "forecast_horizon": row.forecast_horizon,
            "rmse_window": row.rmse_window,
            "percentile_target": row.percentile_target,
        }
        for row in rows
    ]


def replenishment_decision_rows_to_dataframe(
    rows: Iterable[ReplenishmentDecisionRow],
    *,
    library: str = "pandas",
):
    data = replenishment_decision_rows_to_dicts(rows)
    if library == "pandas":
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required for replenishment_decision_rows_to_dataframe(library='pandas')."
            ) from exc
        return pd.DataFrame(data)
    if library == "polars":
        try:
            import polars as pl  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "polars is required for replenishment_decision_rows_to_dataframe(library='polars')."
            ) from exc
        return pl.DataFrame(data)
    raise ValueError("library must be 'pandas' or 'polars'.")


def standard_simulation_rows_from_dataframe(
    df,
    *,
    unique_id_field: str = "unique_id",
    ds_field: str = "ds",
    demand_field: str = "demand",
    history_field: str = "history",
    forecast_field: str = "forecast",
    actuals_field: str = "actuals",
    holding_cost_per_unit_field: str = "holding_cost_per_unit",
    stockout_cost_per_unit_field: str = "stockout_cost_per_unit",
    order_cost_per_order_field: str = "order_cost_per_order",
    lead_time_field: str = "lead_time",
    initial_on_hand_field: str = "initial_on_hand",
    initial_demand_field: str = "initial_demand",
    current_stock_field: str = "current_stock",
    is_forecast_field: str = "is_forecast",
    period_field: str = "period",
    cutoff: int | str | date | datetime | None = None,
    forecast_prefix: str = "forecast_",
) -> list[StandardSimulationRow]:
    """Convert a pandas or polars DataFrame into standard simulation rows."""
    rows = _rows_from_dataframe(df)
    if not rows:
        return []
    fieldnames = list(rows[0].keys())
    has_demand = demand_field in fieldnames
    has_history = history_field in fieldnames
    has_actuals = actuals_field in fieldnames
    required = [
        unique_id_field,
        ds_field,
        forecast_field,
        holding_cost_per_unit_field,
        stockout_cost_per_unit_field,
        order_cost_per_order_field,
        lead_time_field,
        current_stock_field,
    ]
    _validate_required_columns(
        fieldnames,
        required_fields=required,
        require_any_of=[initial_on_hand_field, initial_demand_field],
        context="standard simulation DataFrame",
    )
    if not has_demand and not has_history and not has_actuals:
        raise ValueError(
            "DataFrame must include demand, history, or actuals columns to build simulation rows."
        )
    if not has_actuals and not has_history:
        raise ValueError(
            "DataFrame must include actuals or history columns to build simulation rows."
        )

    parsed_rows: list[StandardSimulationRow] = []
    for row in rows:
        forecast_percentiles = {
            key[len(forecast_prefix) :]: int(value)
            for key, value in row.items()
            if key.startswith(forecast_prefix)
            and key != forecast_field
            and not _is_missing(value)
        }
        initial_on_hand = _coalesce_value(
            row, initial_on_hand_field, initial_demand_field
        )
        if initial_on_hand is None:
            raise ValueError(
                "Initial on-hand inventory is required (initial_on_hand or initial_demand)."
            )
        current_stock_value = _coalesce_value(row, current_stock_field)
        if current_stock_value is None:
            current_stock_value = initial_on_hand
        is_forecast_value = _derive_is_forecast(
            row,
            ds_field=ds_field,
            is_forecast_field=is_forecast_field,
            period_field=period_field,
            actuals_field=actuals_field,
            history_field=history_field,
            cutoff=cutoff,
        )
        demand_value = _coalesce_value(row, demand_field, history_field)
        if demand_value is None:
            demand_value = _coalesce_value(row, actuals_field)
        if demand_value is None:
            demand_value = _coalesce_value(row, forecast_field)
        if demand_value is None:
            raise ValueError("Demand values are required for all periods.")
        actuals_value = _coalesce_value(row, actuals_field, history_field)
        if actuals_value is None and not is_forecast_value:
            raise ValueError("Actuals values are required for backtest periods.")
        parsed_rows.append(
            StandardSimulationRow(
                unique_id=str(row[unique_id_field]),
                ds=_normalize_ds(row[ds_field]),
                demand=int(demand_value),
                forecast=int(row[forecast_field]),
                actuals=actuals_value,
                holding_cost_per_unit=float(row[holding_cost_per_unit_field]),
                stockout_cost_per_unit=float(row[stockout_cost_per_unit_field]),
                order_cost_per_order=float(row[order_cost_per_order_field]),
                lead_time=int(row[lead_time_field]),
                initial_on_hand=int(initial_on_hand),
                current_stock=int(current_stock_value),
                forecast_percentiles=forecast_percentiles,
                is_forecast=is_forecast_value,
            )
        )
    return parsed_rows


def write_standard_simulation_rows_to_csv(
    path: str,
    rows: Iterable[StandardSimulationRow],
    *,
    forecast_prefix: str = "forecast_",
) -> None:
    """Write standard simulation rows to a CSV that matches the README schema."""
    rows_list = list(rows)
    percentile_labels: list[str] = []
    for row in rows_list:
        for label in row.forecast_percentiles:
            if label not in percentile_labels:
                percentile_labels.append(label)
    fieldnames = [
        "unique_id",
        "ds",
        "demand",
        "forecast",
        "actuals",
        "holding_cost_per_unit",
        "stockout_cost_per_unit",
        "order_cost_per_order",
        "lead_time",
        "initial_on_hand",
        "current_stock",
        "is_forecast",
    ] + [f"{forecast_prefix}{label}" for label in percentile_labels]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in standard_simulation_rows_to_dicts(
            rows_list, forecast_prefix=forecast_prefix
        ):
            if _is_missing(entry.get("actuals")):
                entry["actuals"] = ""
            writer.writerow(entry)


def iter_point_forecast_rows_from_csv(
    path: str,
    *,
    unique_id_field: str = "unique_id",
    period_field: str = "period",
    demand_field: str = "demand",
    forecast_field: str = "forecast",
    actual_field: str = "actual",
) -> Iterator[PointForecastRow]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[
                unique_id_field,
                period_field,
                demand_field,
                forecast_field,
                actual_field,
            ],
            context="point-forecast CSV",
        )
        for row in reader:
            yield PointForecastRow(
                unique_id=row[unique_id_field],
                period=int(row[period_field]),
                demand=int(row[demand_field]),
                forecast=int(row[forecast_field]),
                actual=int(row[actual_field]),
            )


def iter_percentile_forecast_rows_from_csv(
    path: str,
    *,
    unique_id_field: str = "unique_id",
    period_field: str = "period",
    demand_field: str = "demand",
    target_field: str = "target",
    forecast_field: str = "forecast",
) -> Iterator[PercentileForecastRow]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[
                unique_id_field,
                period_field,
                demand_field,
                target_field,
                forecast_field,
            ],
            context="percentile-forecast CSV",
        )
        for row in reader:
            yield PercentileForecastRow(
                unique_id=row[unique_id_field],
                period=int(row[period_field]),
                demand=int(row[demand_field]),
                target=row[target_field],
                forecast=int(row[forecast_field]),
            )


def iter_standard_simulation_rows_from_csv(
    path: str,
    *,
    unique_id_field: str = "unique_id",
    ds_field: str = "ds",
    demand_field: str = "demand",
    forecast_field: str = "forecast",
    actuals_field: str = "actuals",
    holding_cost_per_unit_field: str = "holding_cost_per_unit",
    stockout_cost_per_unit_field: str = "stockout_cost_per_unit",
    order_cost_per_order_field: str = "order_cost_per_order",
    lead_time_field: str = "lead_time",
    initial_on_hand_field: str = "initial_on_hand",
    initial_demand_field: str = "initial_demand",
    current_stock_field: str = "current_stock",
    is_forecast_field: str = "is_forecast",
    forecast_prefix: str = "forecast_",
) -> Iterator[StandardSimulationRow]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[
                unique_id_field,
                ds_field,
                demand_field,
                forecast_field,
                actuals_field,
                holding_cost_per_unit_field,
                stockout_cost_per_unit_field,
                order_cost_per_order_field,
                lead_time_field,
                current_stock_field,
            ],
            context="standard simulation CSV",
        )
        _validate_required_columns(
            reader.fieldnames,
            required_fields=[],
            require_any_of=[initial_on_hand_field, initial_demand_field],
            context="standard simulation CSV",
        )
        for row in reader:
            forecast_percentiles = {
                key[len(forecast_prefix) :]: int(value)
                for key, value in row.items()
                if key.startswith(forecast_prefix)
                and key != forecast_field
                and value != ""
            }
            _validate_matching_initial_inventory(
                row,
                initial_on_hand_field=initial_on_hand_field,
                initial_demand_field=initial_demand_field,
            )
            initial_on_hand = _coalesce_row_value(
                row, initial_on_hand_field, initial_demand_field
            )
            if initial_on_hand is None:
                raise ValueError(
                    "Initial on-hand inventory is required (initial_on_hand or initial_demand)."
                )
            current_stock_value = _coalesce_row_value(row, current_stock_field)
            if current_stock_value is None:
                current_stock_value = initial_on_hand
            is_forecast_value = False
            if is_forecast_field in row and row[is_forecast_field] != "":
                is_forecast_value = _parse_bool(
                    row[is_forecast_field], field=is_forecast_field
                )
            actuals_value = _parse_optional_int(row.get(actuals_field, ""))
            if actuals_value is None and not is_forecast_value:
                raise ValueError("Actuals values are required for backtest periods.")
            yield StandardSimulationRow(
                unique_id=row[unique_id_field],
                ds=row[ds_field],
                demand=int(row[demand_field]),
                forecast=int(row[forecast_field]),
                actuals=actuals_value,
                holding_cost_per_unit=float(row[holding_cost_per_unit_field]),
                stockout_cost_per_unit=float(row[stockout_cost_per_unit_field]),
                order_cost_per_order=float(row[order_cost_per_order_field]),
                lead_time=int(row[lead_time_field]),
                initial_on_hand=initial_on_hand,
                current_stock=current_stock_value,
                forecast_percentiles=forecast_percentiles,
                is_forecast=is_forecast_value,
            )


def build_point_forecast_article_configs(
    rows: Iterable[PointForecastRow],
    *,
    lead_time: Mapping[str, int] | int,
    initial_on_hand: Mapping[str, int] | int,
    service_level_factor: Mapping[str, float] | float,
    service_level_mode: Mapping[str, str] | str | None = None,
    safety_stock_method: Mapping[str, str] | str | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    rmse_window: Mapping[str, int] | int | None = None,
    policy_mode: str = "base_stock",
    holding_cost_per_unit: Mapping[str, float] | float = 0.0,
    stockout_cost_per_unit: Mapping[str, float] | float = 0.0,
    order_cost_per_order: Mapping[str, float] | float = 0.0,
    order_cost_per_unit: Mapping[str, float] | float = 0.0,
) -> dict[str, ArticleSimulationConfig]:
    # service_level_mode/rmse_window are accepted for signature compatibility
    # but not honored: safety_stock selection here only maps
    # safety_stock_method to SqrtHorizonSafetyStock/KRmseSafetyStock/
    # KMaeSafetyStock (Task 8), which always compute error from the full
    # actuals-vs-forecast history with no fixed-window or fill-rate-mode
    # option. See task-11-report.md concerns.
    grouped: dict[str, dict[int, PointForecastRow]] = defaultdict(dict)
    for row in rows:
        if row.period < 0:
            raise ValueError("Period cannot be negative.")
        article_rows = grouped[row.unique_id]
        if row.period in article_rows:
            raise ValueError(
                f"Duplicate period {row.period} for unique_id '{row.unique_id}'."
            )
        article_rows[row.period] = row

    configs: dict[str, ArticleSimulationConfig] = {}
    for unique_id, period_rows in grouped.items():
        periods = _validate_periods(unique_id, period_rows)
        demand = [period_rows[index].demand for index in range(periods)]
        forecast = [period_rows[index].forecast for index in range(periods)]
        actuals = [period_rows[index].actual for index in range(periods)]
        safety_method_value = _resolve_optional_value(
            safety_stock_method, unique_id, "safety_stock_method"
        )
        if safety_method_value is not None and not isinstance(safety_method_value, str):
            raise TypeError(
                "safety_stock_method must be a string or mapping of strings."
            )
        article_lead_time = _resolve_value(lead_time, unique_id, "lead_time")
        article_review_period = _resolve_optional_value(
            review_period, unique_id, "review_period"
        ) or 1
        article_forecast_horizon = _resolve_optional_value(
            forecast_horizon, unique_id, "forecast_horizon"
        ) or 1
        factor = _resolve_value(service_level_factor, unique_id, "service_level_factor")
        if policy_mode == "rop":
            trigger = ReorderPointTrigger()
        elif policy_mode == "base_stock":
            trigger = OrderUpToTrigger()
        else:
            raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
        policy = ReplenishmentPolicy(
            forecast=TimeSeries.from_values(forecast),
            actuals=TimeSeries.from_values(actuals),
            safety_stock=_safety_stock_builder_for_method(safety_method_value, factor)(),
            trigger=trigger,
            lead_time=article_lead_time,
            review_period=article_review_period,
            forecast_horizon=article_forecast_horizon,
        )
        configs[unique_id] = ArticleSimulationConfig(
            periods=periods,
            demand=demand,
            initial_on_hand=_resolve_value(
                initial_on_hand, unique_id, "initial_on_hand"
            ),
            lead_time=article_lead_time,
            policy=policy,
            holding_cost_per_unit=_resolve_value(
                holding_cost_per_unit, unique_id, "holding_cost_per_unit"
            ),
            stockout_cost_per_unit=_resolve_value(
                stockout_cost_per_unit, unique_id, "stockout_cost_per_unit"
            ),
            order_cost_per_order=_resolve_value(
                order_cost_per_order, unique_id, "order_cost_per_order"
            ),
            order_cost_per_unit=_resolve_value(
                order_cost_per_unit, unique_id, "order_cost_per_unit"
            ),
        )
    return configs


def build_percentile_forecast_candidates(
    rows: Iterable[PercentileForecastRow],
    *,
    lead_time: Mapping[str, int] | int,
    initial_on_hand: Mapping[str, int] | int,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    holding_cost_per_unit: Mapping[str, float] | float = 0.0,
    stockout_cost_per_unit: Mapping[str, float] | float = 0.0,
    order_cost_per_order: Mapping[str, float] | float = 0.0,
    order_cost_per_unit: Mapping[str, float] | float = 0.0,
) -> dict[str, ForecastCandidatesConfig]:
    demand_by_article: dict[str, dict[int, int]] = defaultdict(dict)
    forecast_by_article: dict[str, dict[float | str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in rows:
        if row.period < 0:
            raise ValueError("Period cannot be negative.")
        demand_rows = demand_by_article[row.unique_id]
        if row.period in demand_rows and demand_rows[row.period] != row.demand:
            raise ValueError(
                "Demand must be consistent across targets for each period."
            )
        demand_rows[row.period] = row.demand
        target_rows = forecast_by_article[row.unique_id][row.target]
        if row.period in target_rows:
            raise ValueError(
                f"Duplicate period {row.period} for unique_id '{row.unique_id}' target '{row.target}'."
            )
        target_rows[row.period] = row.forecast

    configs: dict[str, ForecastCandidatesConfig] = {}
    for unique_id, period_rows in demand_by_article.items():
        periods = _validate_periods(unique_id, period_rows)
        demand = [period_rows[index] for index in range(periods)]
        article_lead_time = _resolve_value(lead_time, unique_id, "lead_time")
        article_review_period = _resolve_optional_value(
            review_period, unique_id, "review_period"
        ) or 1
        article_forecast_horizon = _resolve_optional_value(
            forecast_horizon, unique_id, "forecast_horizon"
        ) or 1
        candidate_policies: dict[float | str, ReplenishmentPolicy] = {}
        for target, target_rows in forecast_by_article[unique_id].items():
            _validate_periods(unique_id, target_rows)
            series = [target_rows[index] for index in range(periods)]
            candidate_policies[target] = ReplenishmentPolicy.order_up_to(
                forecast=TimeSeries.from_values(series),
                safety_stock=NullSafetyStockStrategy(),
                lead_time=article_lead_time,
                review_period=article_review_period,
                forecast_horizon=article_forecast_horizon,
            )

        configs[unique_id] = ForecastCandidatesConfig(
            periods=periods,
            demand=demand,
            initial_on_hand=_resolve_value(
                initial_on_hand, unique_id, "initial_on_hand"
            ),
            lead_time=article_lead_time,
            review_period=article_review_period,
            forecast_horizon=article_forecast_horizon,
            candidate_policies=candidate_policies,
            holding_cost_per_unit=_resolve_value(
                holding_cost_per_unit, unique_id, "holding_cost_per_unit"
            ),
            stockout_cost_per_unit=_resolve_value(
                stockout_cost_per_unit, unique_id, "stockout_cost_per_unit"
            ),
            order_cost_per_order=_resolve_value(
                order_cost_per_order, unique_id, "order_cost_per_order"
            ),
            order_cost_per_unit=_resolve_value(
                order_cost_per_unit, unique_id, "order_cost_per_unit"
            ),
        )

    return configs


def build_point_forecast_article_configs_from_standard_rows(
    rows: Iterable[StandardSimulationRow],
    *,
    service_level_factor: Mapping[str, float] | float,
    service_level_mode: Mapping[str, str] | str | None = None,
    safety_stock_method: Mapping[str, str] | str | None = None,
    fixed_rmse: Mapping[str, float] | float | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    rmse_window: Mapping[str, int] | int | None = None,
    use_current_stock: bool | None = None,
    actuals_override: Mapping[str, Iterable[int]] | None = None,
    policy_mode: str = "base_stock",
) -> dict[str, ArticleSimulationConfig]:
    # fixed_rmse/rmse_window are accepted for signature compatibility but not
    # honored: SqrtHorizonSafetyStock/KRmseSafetyStock/KMaeSafetyStock (Task 8)
    # always compute error from the full actuals-vs-forecast history, with no
    # fixed-override or windowed-RMSE option. See task-11-report.md concerns.
    grouped = _group_standard_rows(rows)
    configs: dict[str, ArticleSimulationConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        current_stock = _ensure_constant(unique_id, ordered, "current_stock")
        if use_current_stock is None:
            use_current = all(row.is_forecast for row in ordered)
        else:
            use_current = use_current_stock
        starting_stock = current_stock if use_current else initial_on_hand
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        forecast = [row.forecast for row in ordered]
        if actuals_override is None:
            actuals = _trim_actuals_series(unique_id, ordered)
        else:
            if unique_id not in actuals_override:
                raise ValueError(
                    f"Missing actuals override for unique_id '{unique_id}'."
                )
            actuals = list(actuals_override[unique_id])
        safety_method_value = _resolve_optional_value(
            safety_stock_method, unique_id, "safety_stock_method"
        )
        if safety_method_value is not None and not isinstance(safety_method_value, str):
            raise TypeError(
                "safety_stock_method must be a string or mapping of strings."
            )
        factor = _resolve_value(service_level_factor, unique_id, "service_level_factor")
        article_review_period = _resolve_optional_value(
            review_period, unique_id, "review_period"
        ) or 1
        article_forecast_horizon = _resolve_optional_value(
            forecast_horizon, unique_id, "forecast_horizon"
        ) or 1
        if policy_mode == "rop":
            trigger = ReorderPointTrigger()
        elif policy_mode == "base_stock":
            trigger = OrderUpToTrigger()
        else:
            raise ValueError("policy_mode must be 'base_stock' or 'rop'.")
        policy = ReplenishmentPolicy(
            forecast=TimeSeries.from_values(forecast),
            actuals=TimeSeries.from_values(actuals),
            safety_stock=_safety_stock_builder_for_method(safety_method_value, factor)(),
            trigger=trigger,
            lead_time=lead_time,
            review_period=article_review_period,
            forecast_horizon=article_forecast_horizon,
        )
        configs[unique_id] = ArticleSimulationConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=starting_stock,
            lead_time=lead_time,
            policy=policy,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
    return configs


def build_lead_time_forecast_article_configs_from_standard_rows(
    rows: Iterable[StandardSimulationRow],
    *,
    service_level_factor: Mapping[str, float] | float,
    service_level_mode: Mapping[str, str] | str | None = None,
    safety_stock_method: Mapping[str, str] | str | None = None,
    fixed_rmse: Mapping[str, float] | float | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
    rmse_window: Mapping[str, int] | int | None = None,
    use_current_stock: bool | None = None,
    actuals_override: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, ArticleSimulationConfig]:
    # janrth's LeadTimeForecastOptimizationPolicy used the same
    # protection_horizon = lead_time + forecast_horizon scaling as
    # PointForecastOptimizationPolicy's order_quantity_for/safety-stock math
    # -- SqrtHorizonSafetyStock (Task 8) already combines lead_time + horizon
    # this way internally, so no extra forecast_horizon adjustment is needed
    # here beyond passing lead_time and forecast_horizon straight through.
    #
    # service_level_mode/fixed_rmse/rmse_window are accepted for signature
    # compatibility but not honored: SqrtHorizonSafetyStock/KRmseSafetyStock/
    # KMaeSafetyStock (Task 8) always compute error from the full
    # actuals-vs-forecast history, with no fixed-override, windowed-RMSE, or
    # fill-rate-mode option. See task-11-report.md concerns.
    grouped = _group_standard_rows(rows)
    configs: dict[str, ArticleSimulationConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        current_stock = _ensure_constant(unique_id, ordered, "current_stock")
        if use_current_stock is None:
            use_current = all(row.is_forecast for row in ordered)
        else:
            use_current = use_current_stock
        starting_stock = current_stock if use_current else initial_on_hand
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        forecast = [row.forecast for row in ordered]
        if actuals_override is None:
            actuals = _trim_actuals_series(unique_id, ordered)
        else:
            if unique_id not in actuals_override:
                raise ValueError(
                    f"Missing actuals override for unique_id '{unique_id}'."
                )
            actuals = list(actuals_override[unique_id])
        safety_method_value = _resolve_optional_value(
            safety_stock_method, unique_id, "safety_stock_method"
        )
        if safety_method_value is not None and not isinstance(safety_method_value, str):
            raise TypeError(
                "safety_stock_method must be a string or mapping of strings."
            )
        factor = _resolve_value(service_level_factor, unique_id, "service_level_factor")
        article_review_period = _resolve_optional_value(
            review_period, unique_id, "review_period"
        ) or 1
        article_forecast_horizon = _resolve_optional_value(
            forecast_horizon, unique_id, "forecast_horizon"
        ) or 1
        policy = ReplenishmentPolicy(
            forecast=TimeSeries.from_values(forecast),
            actuals=TimeSeries.from_values(actuals),
            safety_stock=_safety_stock_builder_for_method(safety_method_value, factor)(),
            trigger=OrderUpToTrigger(),
            lead_time=lead_time,
            review_period=article_review_period,
            forecast_horizon=article_forecast_horizon,
        )
        configs[unique_id] = ArticleSimulationConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=starting_stock,
            lead_time=lead_time,
            policy=policy,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
    return configs


@dataclass(frozen=True)
class _FrozenSafetyStock:
    """A safety-stock strategy that always returns a precomputed value.
    Used to carry a correctly-computed backtest-window safety-stock number
    into an evaluation-window policy, without ever pairing that window's
    forecast against a different window's actuals in one TimeSeries pair
    (the bug this replaces)."""
    value: float

    def compute(self, *, forecast, actuals, period, lead_time, horizon, service_level_factor) -> float:
        return self.value


def optimize_point_forecast_policy_and_simulate_actuals(
    backtest_rows: Iterable[StandardSimulationRow],
    evaluation_rows: Iterable[StandardSimulationRow],
    candidate_factors: Iterable[float],
    *,
    use_current_stock: bool | None = None,
    service_level_mode: str | None = None,
    safety_stock_method: str | None = None,
) -> tuple[
    dict[str, CalibrationResult],
    dict[str, SimulationResult],
    list[ReplenishmentDecisionRow],
]:
    """Calibrate each article's safety-stock factor against backtest rows via
    calibration.optimize (Task 10), then simulate the evaluation rows with the
    winning factor. Adapted from janrth's
    optimize_service_level_factors + simulate_replenishment_for_articles
    pipeline (neither of which this repo has ported, see module docstring):
    calibration.optimize already does a train/validation split that janrth's
    single-window optimizer lacked, so it's reused directly per article
    rather than re-implemented here."""
    backtest_grouped = _group_standard_rows(backtest_rows)
    eval_grouped = _group_standard_rows(evaluation_rows)
    backtest_actuals = _actuals_by_article(backtest_rows)

    optimized: dict[str, CalibrationResult] = {}
    eval_simulations: dict[str, SimulationResult] = {}
    eval_decisions: list[ReplenishmentDecisionRow] = []

    for unique_id, ds_rows in backtest_grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        demand = [row.demand for row in ordered]
        forecast_values = [row.forecast for row in ordered]
        actuals_values = backtest_actuals[unique_id]

        def candidate_builder(
            factor, forecast_values=forecast_values, actuals_values=actuals_values,
            lead_time=lead_time,
        ) -> ReplenishmentPolicy:
            return ReplenishmentPolicy.order_up_to(
                forecast=TimeSeries.from_values(forecast_values),
                actuals=TimeSeries.from_values(actuals_values),
                safety_stock=_safety_stock_builder_for_method(safety_stock_method, factor)(),
                lead_time=lead_time,
            )

        result = calibration_optimize(
            candidate_builder=candidate_builder,
            candidate_values=list(candidate_factors),
            periods=len(ordered),
            demand=demand,
            initial_on_hand=initial_on_hand,
            lead_time=lead_time,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
        optimized[unique_id] = result

        if unique_id not in eval_grouped:
            continue
        eval_ordered = _order_rows_by_ds(unique_id, eval_grouped[unique_id])
        eval_current_stock = _ensure_constant(unique_id, eval_ordered, "current_stock")
        eval_initial_on_hand = _ensure_constant(unique_id, eval_ordered, "initial_on_hand")
        if use_current_stock is None:
            use_current = all(row.is_forecast for row in eval_ordered)
        else:
            use_current = use_current_stock
        eval_starting_stock = eval_current_stock if use_current else eval_initial_on_hand
        eval_demand = [row.demand for row in eval_ordered]
        eval_forecast_values = [row.forecast for row in eval_ordered]
        # Compute the safety-stock buffer correctly from the backtest window
        # (forecast and actuals genuinely aligned, same window, same indices),
        # then carry that single number forward as a frozen constant for the
        # eval simulation -- rather than pairing eval-window forecast against
        # backtest-window actuals in one policy (which silently computed a
        # meaningless residual between two different points in time).
        backtest_policy = candidate_builder(result.best_value)
        frozen_safety_stock_value = backtest_policy.safety_stock.compute(
            forecast=backtest_policy.forecast, actuals=backtest_policy.actuals,
            period=len(ordered), lead_time=lead_time, horizon=1,
            service_level_factor=result.best_value,
        )
        eval_policy = ReplenishmentPolicy.order_up_to(
            forecast=TimeSeries.from_values(eval_forecast_values),
            safety_stock=_FrozenSafetyStock(value=frozen_safety_stock_value),
            lead_time=lead_time,
        )
        simulation = simulate_replenishment(
            periods=len(eval_ordered),
            demand=eval_demand,
            initial_on_hand=eval_starting_stock,
            lead_time=lead_time,
            policy=eval_policy,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
        eval_simulations[unique_id] = simulation
        for row, snapshot in zip(eval_ordered, simulation.snapshots):
            eval_decisions.append(
                ReplenishmentDecisionRow(
                    unique_id=unique_id,
                    ds=row.ds,
                    quantity=snapshot.order_placed,
                    demand=snapshot.demand,
                    incoming_stock=snapshot.received,
                    starting_on_hand=snapshot.starting_on_hand,
                    ending_on_hand=snapshot.ending_on_hand,
                    on_order=snapshot.on_order,
                    backorders=snapshot.backorders,
                    sigma=result.best_value,
                    service_level_mode=service_level_mode,
                )
            )
    return optimized, eval_simulations, eval_decisions


def build_percentile_forecast_candidates_from_standard_rows(
    rows: Iterable[StandardSimulationRow],
    *,
    include_mean: bool = False,
    use_current_stock: bool | None = None,
    review_period: Mapping[str, int] | int | None = None,
    forecast_horizon: Mapping[str, int] | int | None = None,
) -> dict[str, ForecastCandidatesConfig]:
    grouped = _group_standard_rows(rows)
    configs: dict[str, ForecastCandidatesConfig] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        lead_time = _ensure_constant(unique_id, ordered, "lead_time")
        initial_on_hand = _ensure_constant(unique_id, ordered, "initial_on_hand")
        current_stock = _ensure_constant(unique_id, ordered, "current_stock")
        if use_current_stock is None:
            use_current = all(row.is_forecast for row in ordered)
        else:
            use_current = use_current_stock
        starting_stock = current_stock if use_current else initial_on_hand
        holding_cost = _ensure_constant(unique_id, ordered, "holding_cost_per_unit")
        stockout_cost = _ensure_constant(unique_id, ordered, "stockout_cost_per_unit")
        order_cost = _ensure_constant(unique_id, ordered, "order_cost_per_order")
        demand = [row.demand for row in ordered]
        targets = _validate_percentile_targets(unique_id, ordered)
        article_review_period = _resolve_optional_value(
            review_period, unique_id, "review_period"
        ) or 1
        article_forecast_horizon = _resolve_optional_value(
            forecast_horizon, unique_id, "forecast_horizon"
        ) or 1
        candidate_policies: dict[float | str, ReplenishmentPolicy] = {}
        for target in targets:
            series = [row.forecast_percentiles[target] for row in ordered]
            candidate_policies[target] = ReplenishmentPolicy.order_up_to(
                forecast=TimeSeries.from_values(series),
                safety_stock=NullSafetyStockStrategy(),
                lead_time=lead_time,
                review_period=article_review_period,
                forecast_horizon=article_forecast_horizon,
            )
        if include_mean and "mean" not in candidate_policies:
            series = [row.forecast for row in ordered]
            candidate_policies["mean"] = ReplenishmentPolicy.order_up_to(
                forecast=TimeSeries.from_values(series),
                safety_stock=NullSafetyStockStrategy(),
                lead_time=lead_time,
                review_period=article_review_period,
                forecast_horizon=article_forecast_horizon,
            )
        configs[unique_id] = ForecastCandidatesConfig(
            periods=len(ordered),
            demand=demand,
            initial_on_hand=starting_stock,
            lead_time=lead_time,
            review_period=article_review_period,
            forecast_horizon=article_forecast_horizon,
            candidate_policies=candidate_policies,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            order_cost_per_order=order_cost,
        )
    return configs


def split_standard_simulation_rows(
    rows,
    cutoff: int | str | date | datetime | None = None,
) -> tuple[list[StandardSimulationRow], list[StandardSimulationRow]]:
    """Split standard rows into backtest and forecast partitions."""
    if hasattr(rows, "to_dicts") or hasattr(rows, "to_dict"):
        rows = standard_simulation_rows_from_dataframe(rows, cutoff=cutoff)
    backtest_rows: list[StandardSimulationRow] = []
    forecast_rows: list[StandardSimulationRow] = []
    for row in rows:
        if row.is_forecast:
            forecast_rows.append(row)
        else:
            backtest_rows.append(row)
    return backtest_rows, forecast_rows


def _resolve_value(
    value: Mapping[str, int | float] | int | float, unique_id: str, name: str
) -> int | float:
    if isinstance(value, Mapping):
        if unique_id not in value:
            raise ValueError(f"Missing {name} for unique_id '{unique_id}'.")
        return value[unique_id]
    return value


def _resolve_optional_value(
    value: Mapping[str, int | float | str] | int | float | str | None,
    unique_id: str,
    name: str,
) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if unique_id not in value:
            raise ValueError(f"Missing {name} for unique_id '{unique_id}'.")
        return value[unique_id]
    return value


def _validate_periods(unique_id: str, period_rows: Mapping[int, object]) -> int:
    if not period_rows:
        raise ValueError(f"No periods provided for unique_id '{unique_id}'.")
    max_period = max(period_rows)
    expected = set(range(max_period + 1))
    missing = expected.difference(period_rows.keys())
    if missing:
        missing_display = ", ".join(str(period) for period in sorted(missing))
        raise ValueError(
            f"Missing periods for unique_id '{unique_id}': {missing_display}."
        )
    return max_period + 1


def _coalesce_row_value(row: Mapping[str, str], *fields: str) -> int | None:
    for field in fields:
        if field in row and row[field] != "":
            return int(row[field])
    return None


def _coalesce_value(row: Mapping[str, object], *fields: str) -> int | float | None:
    for field in fields:
        if field in row and not _is_missing(row[field]):
            return row[field]  # type: ignore[return-value]
    return None


def _group_standard_rows(
    rows: Iterable[StandardSimulationRow],
) -> dict[str, dict[str, StandardSimulationRow]]:
    grouped: dict[str, dict[str, StandardSimulationRow]] = defaultdict(dict)
    for row in rows:
        article_rows = grouped[row.unique_id]
        if row.ds in article_rows:
            raise ValueError(f"Duplicate ds '{row.ds}' for unique_id '{row.unique_id}'.")
        article_rows[row.ds] = row
    return grouped


def _order_rows_by_ds(
    unique_id: str, ds_rows: Mapping[str, StandardSimulationRow]
) -> list[StandardSimulationRow]:
    if not ds_rows:
        raise ValueError(f"No periods provided for unique_id '{unique_id}'.")
    return [ds_rows[ds] for ds in sorted(ds_rows)]


def _ensure_constant(
    unique_id: str, rows: Iterable[StandardSimulationRow], field: str
) -> int | float:
    values = {getattr(row, field) for row in rows}
    if len(values) != 1:
        value_list = ", ".join(str(value) for value in sorted(values))
        raise ValueError(
            f"{field} must be constant for unique_id '{unique_id}': {value_list}."
        )
    return values.pop()


def _validate_percentile_targets(
    unique_id: str, rows: Iterable[StandardSimulationRow]
) -> list[str]:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError(f"No periods provided for unique_id '{unique_id}'.")
    targets = set(rows_list[0].forecast_percentiles.keys())
    if not targets:
        raise ValueError(f"No percentile forecasts provided for unique_id '{unique_id}'.")
    for row in rows_list[1:]:
        if set(row.forecast_percentiles.keys()) != targets:
            raise ValueError(
                f"Percentile forecasts must be consistent across ds for unique_id '{unique_id}'."
            )
    return sorted(targets)


def _validate_required_columns(
    fieldnames: list[str] | None,
    *,
    required_fields: Iterable[str],
    require_any_of: Iterable[str] | None = None,
    context: str,
) -> None:
    if not fieldnames:
        warnings.warn(f"Missing header row for {context}.", stacklevel=2)
        raise ValueError(f"{context} is missing a header row.")
    field_set = set(fieldnames)
    missing_required = [field for field in required_fields if field not in field_set]
    if require_any_of:
        alternatives = [field for field in require_any_of if field in field_set]
        if not alternatives:
            missing_required.extend(require_any_of)
    if missing_required:
        missing_display = ", ".join(missing_required)
        warnings.warn(
            f"Missing required columns for {context}: {missing_display}.",
            stacklevel=2,
        )
        raise ValueError(
            f"{context} is missing required columns: {missing_display}."
        )


def _validate_matching_initial_inventory(
    row: Mapping[str, str],
    *,
    initial_on_hand_field: str,
    initial_demand_field: str,
) -> None:
    if (
        initial_on_hand_field in row
        and initial_demand_field in row
        and row[initial_on_hand_field] != ""
        and row[initial_demand_field] != ""
    ):
        initial_on_hand = int(row[initial_on_hand_field])
        initial_demand = int(row[initial_demand_field])
        if initial_on_hand != initial_demand:
            warnings.warn(
                "Initial inventory mismatch between "
                f"{initial_on_hand_field} ({initial_on_hand}) and "
                f"{initial_demand_field} ({initial_demand}).",
                stacklevel=2,
            )
            raise ValueError(
                f"{initial_on_hand_field} and {initial_demand_field} must match when both are provided."
            )


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "t"}:
        return True
    if normalized in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError(f"Invalid boolean value for {field}: {value!r}.")


def _parse_optional_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _rows_from_dataframe(df) -> list[dict[str, object]]:
    if hasattr(df, "to_dicts"):
        return df.to_dicts()  # type: ignore[no-any-return]
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")  # type: ignore[no-any-return]
    raise TypeError("df must be a pandas or polars DataFrame.")


def _normalize_ds(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _derive_is_forecast(
    row: Mapping[str, object],
    *,
    ds_field: str,
    is_forecast_field: str,
    period_field: str,
    actuals_field: str,
    history_field: str,
    cutoff: int | str | date | datetime | None,
) -> bool:
    if is_forecast_field in row and not _is_missing(row[is_forecast_field]):
        value = row[is_forecast_field]
        if isinstance(value, bool):
            return value
        return _parse_bool(str(value), field=is_forecast_field)
    if cutoff is None:
        return _is_missing(row.get(actuals_field)) and _is_missing(row.get(history_field))
    if isinstance(cutoff, int):
        if period_field not in row or _is_missing(row[period_field]):
            raise ValueError(
                "period field is required when using an integer cutoff for DataFrame inputs."
            )
        return int(row[period_field]) > cutoff
    ds_value = row.get(ds_field)
    if ds_value is None:
        return False
    if isinstance(ds_value, (date, datetime)):
        cutoff_value = cutoff
        if isinstance(cutoff, str):
            cutoff_value = _try_parse_date(cutoff)
        if isinstance(cutoff_value, datetime):
            cutoff_value = cutoff_value.date()
        if isinstance(cutoff_value, date):
            if isinstance(ds_value, datetime):
                ds_value = ds_value.date()
            return ds_value > cutoff_value
        return str(ds_value) > str(cutoff)
    return str(ds_value) > str(cutoff)


def _try_parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _rmse_from_series(actuals: list[int], forecast: list[int]) -> float:
    max_index = min(len(actuals), len(forecast))
    if max_index <= 0:
        return 0.0
    errors = [actuals[index] - forecast[index] for index in range(max_index)]
    if not errors:
        return 0.0
    if len(errors) == 1:
        return abs(errors[0])
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def _aggregate_series(
    series: list[int], *, periods: int, window: int, extend_last: bool = False,
) -> list[int]:
    """Ported verbatim from janrth's aggregation.aggregate_series (pure,
    no policy dependency) -- only the two aggregation helpers actually used
    by compute_backtest_rmse_by_article below are ported, not the rest of
    aggregation.py (which is entangled with the old per-article policy
    ecosystem this repo hasn't ported; see module docstring)."""
    if periods <= 0:
        raise ValueError("Periods must be positive.")
    if window <= 0:
        raise ValueError("Aggregation window must be positive.")
    values = list(series)
    if len(values) < periods:
        if not values or not extend_last:
            raise ValueError("Series shorter than periods.")
        values = values + [values[-1]] * (periods - len(values))
    elif len(values) > periods:
        values = values[:periods]
    return [sum(values[index : index + window]) for index in range(0, periods, window)]


def _aggregate_lead_time(lead_time: int, window: int) -> int:
    """Ported verbatim from janrth's aggregation.aggregate_lead_time."""
    if lead_time < 0:
        raise ValueError("Lead time cannot be negative.")
    if window <= 0:
        raise ValueError("Aggregation window must be positive.")
    return math.ceil(lead_time / window)


def compute_backtest_rmse_by_article(
    rows: Iterable[StandardSimulationRow],
    *,
    aggregation_window: Mapping[str, int] | int = 1,
    rmse_window: Mapping[str, int] | int | None = None,
) -> dict[str, float]:
    """Compute fixed RMSE per article using backtest rows only."""
    if hasattr(rows, "to_dicts") or hasattr(rows, "to_dict"):
        rows = standard_simulation_rows_from_dataframe(rows)
    grouped = _group_standard_rows(rows)
    rmse_by_article: dict[str, float] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        backtest_rows = [row for row in ordered if not row.is_forecast]
        if not backtest_rows:
            raise ValueError(
                f"No backtest rows available for unique_id '{unique_id}'."
            )
        forecast_values = [row.forecast for row in backtest_rows]
        actuals_values = [int(row.actuals) for row in backtest_rows]
        if rmse_window is not None:
            window = _resolve_value(rmse_window, unique_id, "rmse_window")
        else:
            window = _resolve_value(
                aggregation_window, unique_id, "aggregation_window"
            )
        if not isinstance(window, int) or window <= 0:
            raise ValueError("aggregation_window must be a positive int.")
        if window > 1:
            forecast_values = _aggregate_series(
                forecast_values,
                periods=len(forecast_values),
                window=window,
                extend_last=True,
            )
            actuals_values = _aggregate_series(
                actuals_values,
                periods=len(actuals_values),
                window=window,
                extend_last=False,
            )
        rmse_by_article[unique_id] = _rmse_from_series(
            actuals_values, forecast_values
        )
    return rmse_by_article


def _trim_actuals_series(
    unique_id: str, rows: Iterable[StandardSimulationRow]
) -> list[int]:
    trimmed: list[int] = []
    seen_missing = False
    for row in rows:
        actuals_value = row.actuals
        if _is_missing(actuals_value):
            seen_missing = True
            continue
        if seen_missing:
            raise ValueError(
                f"Actuals must be present for all backtest periods for unique_id '{unique_id}'."
            )
        trimmed.append(int(actuals_value))
    return trimmed


def _actuals_by_article(
    rows: Iterable[StandardSimulationRow],
) -> dict[str, list[int]]:
    grouped = _group_standard_rows(rows)
    actuals_by_article: dict[str, list[int]] = {}
    for unique_id, ds_rows in grouped.items():
        ordered = _order_rows_by_ds(unique_id, ds_rows)
        actuals_by_article[unique_id] = _trim_actuals_series(unique_id, ordered)
    return actuals_by_article
