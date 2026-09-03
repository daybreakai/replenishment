"""The single unified replenishment policy, composed from a forecast
TimeSeries, a SafetyStockStrategy, and an OrderTrigger. Replaces
janrth's 7 duplicated policy dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

from replenishment.strategies.multiplier import MultiplierSafetyStockStrategy, NullSafetyStockStrategy
from replenishment.strategies.order_trigger import OrderTrigger, OrderUpToTrigger, ReorderPointTrigger
from replenishment.strategies.safety_stock import (
    FillRateSafetyStock, KMaeSafetyStock, KRmseSafetyStock, SqrtHorizonSafetyStock,
)
from replenishment.timeseries import TimeSeries

_REQUIRES_ACTUALS = (SqrtHorizonSafetyStock, KRmseSafetyStock, KMaeSafetyStock, FillRateSafetyStock)


def round_to_moq(qty: int, moq: int) -> int:
    """A positive order rounds UP to the nearest multiple of moq (pack-size
    semantics: "must buy 12" means 12, 24, 36 -- not 13); a non-positive
    quantity ("don't order") is never turned into one."""
    if qty <= 0:
        return 0
    return ((qty + moq - 1) // moq) * moq


def _requires_actuals(strategy) -> bool:
    """True if `strategy` (or anything it wraps, recursively — e.g. a
    DemandBufferDecorator) needs actuals to compute a safety stock."""
    if isinstance(strategy, _REQUIRES_ACTUALS):
        return True
    wrapped = getattr(strategy, "wrapped", None)
    if wrapped is not None:
        return _requires_actuals(wrapped)
    return False


@dataclass(frozen=True)
class ReplenishmentPolicy:
    forecast: TimeSeries
    safety_stock: object  # SafetyStockStrategy
    trigger: OrderTrigger
    actuals: TimeSeries | None = None
    lead_time: int = 0
    review_period: int = 1
    forecast_horizon: int = 1
    moq: int = 1  # order multiple (pack size): rounds any positive order up to a multiple, never creates one

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("lead_time cannot be negative.")
        if self.moq < 1:
            raise ValueError("moq must be at least 1.")
        if self.review_period <= 0:
            raise ValueError("review_period must be positive.")
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if _requires_actuals(self.safety_stock) and self.actuals is None:
            raise ValueError(
                f"{type(self.safety_stock).__name__} requires actuals to compute "
                "forecast error; pass actuals=TimeSeries(...) or choose a "
                "different SafetyStockStrategy."
            )

    def order_quantity_for(self, state) -> int:
        qty = self.trigger.order_quantity(
            inventory_position=state.inventory_position, period=state.period,
            review_period=self.review_period, forecast=self.forecast,
            safety_stock=self._safety_stock_at(state.period),
            lead_time=self.lead_time, forecast_horizon=self.forecast_horizon,
        )
        return round_to_moq(qty, self.moq)

    def status(self, state) -> tuple[float, float]:
        """(reorder_point, order_up_to_target) at `state`, independent of
        whether the trigger has actually fired -- for callers that need to
        know how close an item is to triggering without deciding whether to
        order (e.g. pooled/joint replenishment ranking not-yet-triggered
        items for top-up). Only reorder-point-family triggers expose this;
        an order-up-to trigger has no "not yet triggered" state to describe."""
        if not hasattr(self.trigger, "status"):
            raise AttributeError(
                f"{type(self.trigger).__name__} has no reorder-point status — "
                f"only ReorderPointTrigger/FlatReorderPointTrigger do")
        return self.trigger.status(
            period=state.period, forecast=self.forecast,
            safety_stock=self._safety_stock_at(state.period),
            lead_time=self.lead_time, forecast_horizon=self.forecast_horizon,
        )

    def _safety_stock_at(self, period: int) -> float:
        return self.safety_stock.compute(
            forecast=self.forecast, actuals=self.actuals, period=period,
            lead_time=self.lead_time, horizon=self.forecast_horizon, service_level_factor=1.0,
        )

    @classmethod
    def order_up_to(cls, *, forecast: TimeSeries, safety_stock, actuals: TimeSeries | None = None,
                     lead_time: int = 0, review_period: int = 1, forecast_horizon: int = 1,
                     moq: int = 1) -> "ReplenishmentPolicy":
        return cls(forecast=forecast, actuals=actuals, safety_stock=safety_stock,
                    trigger=OrderUpToTrigger(), lead_time=lead_time,
                    review_period=review_period, forecast_horizon=forecast_horizon, moq=moq)

    @classmethod
    def reorder_point(cls, *, forecast: TimeSeries, safety_stock, actuals: TimeSeries | None = None,
                       lead_time: int = 0, review_period: int = 1, forecast_horizon: int = 1,
                       moq: int = 1) -> "ReplenishmentPolicy":
        return cls(forecast=forecast, actuals=actuals, safety_stock=safety_stock,
                    trigger=ReorderPointTrigger(), lead_time=lead_time,
                    review_period=review_period, forecast_horizon=forecast_horizon, moq=moq)
