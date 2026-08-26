"""Portfolio: the front door for per-item replenishment simulation.

Wraps the io_ builder + per-item simulate loop that every notebook was
hand-rolling::

    port = Portfolio.from_dataframe(df)
    res = port.simulate(factor=1.65, horizon=lead_times)          # order-up-to
    rop = port.simulate(factor=0.8, horizon=lead_times, mode="rop")
    res.summary_frame()       # per-item DataFrame
    res.portfolio_metrics()   # weighted fill, worst month, cost totals

Every knob accepts a scalar or a per-item ``{unique_id: value}`` mapping,
same as the underlying builder.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from replenishment.io_ import (
    ArticleSimulationConfig,
    StandardSimulationRow,
    build_point_forecast_article_configs_from_standard_rows,
    standard_simulation_rows_from_dataframe,
)
from replenishment.simulation import SimulationResult, SimulationSummary


@dataclass(frozen=True)
class PortfolioResult:
    """Per-item SimulationResults plus the aggregate views that matter."""

    results: dict[str, SimulationResult]

    def __getitem__(self, unique_id: str) -> SimulationResult:
        return self.results[unique_id]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results.items())

    @property
    def summaries(self) -> dict[str, SimulationSummary]:
        return {uid: r.summary for uid, r in self.results.items()}

    def summary_frame(self):
        """Per-item DataFrame: fill_rate, costs, avg_on_hand. Index = unique_id."""
        import pandas as pd  # type: ignore

        return pd.DataFrame.from_dict(
            {
                uid: {
                    "fill_rate": s.fill_rate,
                    "total_cost": s.total_cost,
                    "avg_on_hand": s.average_on_hand,
                    "holding_cost": s.holding_cost,
                    "stockout_cost": s.stockout_cost,
                    "ordering_cost": s.ordering_cost,
                }
                for uid, s in self.summaries.items()
            },
            orient="index",
        ).rename_axis("unique_id")

    def portfolio_metrics(self) -> dict[str, float]:
        """Aggregate scorecard.

        weighted_fill (total fulfilled / total demand) and worst_month_fill
        (minimum single-period portfolio fill) are what the customer
        experiences; unweighted mean_fill_rate is kept for comparability but
        can hide volume mix shifts and correlated stockouts.
        """
        summaries = list(self.summaries.values())
        n = len(summaries)
        if n == 0:
            raise ValueError("PortfolioResult is empty.")
        total_demand = sum(s.total_demand for s in summaries)
        demand_by_period: dict[int, int] = {}
        fulfilled_by_period: dict[int, int] = {}
        for r in self.results.values():
            for snap in r.snapshots:
                demand_by_period[snap.period] = demand_by_period.get(snap.period, 0) + snap.demand
                fulfilled_by_period[snap.period] = (
                    fulfilled_by_period.get(snap.period, 0) + snap.demand - snap.backorders
                )
        month_fills = [
            fulfilled_by_period[p] / demand_by_period[p]
            for p in demand_by_period
            if demand_by_period[p] > 0
        ]
        return {
            "mean_fill_rate": sum(s.fill_rate for s in summaries) / n,
            "weighted_fill": (
                sum(s.total_fulfilled for s in summaries) / total_demand if total_demand else 1.0
            ),
            "worst_month_fill": min(month_fills) if month_fills else 1.0,
            "avg_on_hand": sum(s.average_on_hand for s in summaries) / n,
            "total_holding": sum(s.holding_cost for s in summaries),
            "total_stockout": sum(s.stockout_cost for s in summaries),
            "total_ordering": sum(s.ordering_cost for s in summaries),
            "total_cost": sum(s.total_cost for s in summaries),
        }


class Portfolio:
    """A set of items (StandardSimulationRows) ready to simulate under any knobs."""

    def __init__(self, rows: Iterable[StandardSimulationRow]):
        self._rows = list(rows)
        if not self._rows:
            raise ValueError("Portfolio needs at least one row.")

    @classmethod
    def from_dataframe(cls, df, **kwargs) -> "Portfolio":
        """Build from a DataFrame with the standard-row columns (see
        standard_simulation_rows_from_dataframe for accepted kwargs)."""
        return cls(standard_simulation_rows_from_dataframe(df, **kwargs))

    @property
    def rows(self) -> list[StandardSimulationRow]:
        return list(self._rows)

    @property
    def unique_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self._rows:
            seen.setdefault(row.unique_id, None)
        return list(seen)

    def configs(
        self,
        *,
        factor: Mapping[str, float] | float,
        horizon: Mapping[str, int] | int | None = None,
        method: Mapping[str, str] | str = "sqrt_horizon",
        mode: str = "base_stock",
        review_period: Mapping[str, int] | int | None = None,
        moq: Mapping[str, int] | int | None = None,
        use_current_stock: bool | None = None,
        actuals_override: Mapping[str, Iterable[int]] | None = None,
        fixed_error: Mapping[str, float] | float | None = None,
    ) -> dict[str, ArticleSimulationConfig]:
        """Escape hatch: the raw per-item configs (e.g. for calibration.optimize)."""
        return build_point_forecast_article_configs_from_standard_rows(
            self._rows,
            service_level_factor=factor,
            safety_stock_method=method,
            review_period=review_period,
            forecast_horizon=horizon,
            use_current_stock=use_current_stock,
            actuals_override=actuals_override,
            policy_mode=mode,
            moq=moq,
            fixed_error=fixed_error,
        )

    def simulate(self, **knobs) -> PortfolioResult:
        """Build configs and simulate every item. Same knobs as configs():
        factor, horizon, method, mode, review_period, moq, use_current_stock,
        actuals_override, fixed_error. horizon=lead_times is the lead-time-aware policy;
        the default horizon=1 is the naive one-period target."""
        return PortfolioResult(
            {uid: cfg.simulate() for uid, cfg in self.configs(**knobs).items()}
        )
