"""Degradation-ladder selection of a SafetyStockStrategy from data
availability. Mirrors the cascade in Duvo.ai's inventory-optimization
skill (calculate-stock-health.py::calculate_safety_stock): full formula
-> simpler formula -> flat buffer -> never block -- expressed here with
this repo's own SafetyStockStrategy classes instead of a hardcoded
percentage.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from replenishment.strategies.multiplier import MultiplierSafetyStockStrategy, NullSafetyStockStrategy
from replenishment.strategies.safety_stock import FillRateSafetyStock, KRmseSafetyStock, SqrtHorizonSafetyStock


class ResolvedSafetyStock(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: object
    method: str
    degraded: bool
    reason: str | None = None


def resolve_safety_stock_strategy(
    *,
    has_actuals: bool,
    periods_observed: int,
    target_fill_rate: float | None = None,
    factor: float | None = None,
    multiplier: float | None = None,
) -> ResolvedSafetyStock:
    if has_actuals and periods_observed > 0:
        if target_fill_rate is not None:
            try:
                strategy = FillRateSafetyStock(target_fill_rate=target_fill_rate)
                return ResolvedSafetyStock(strategy=strategy, method=type(strategy).__name__, degraded=False)
            except ValueError:
                pass  # invalid target_fill_rate -- fall through to the next rung
        if factor is not None:
            strategy = SqrtHorizonSafetyStock(factor=factor)
            return ResolvedSafetyStock(strategy=strategy, method=type(strategy).__name__, degraded=False)
        strategy = KRmseSafetyStock(factor=1.0)
        return ResolvedSafetyStock(
            strategy=strategy, method=type(strategy).__name__, degraded=True,
            reason="No target_fill_rate or factor given (or target_fill_rate was invalid); using a flat "
                   "K-RMSE buffer (factor=1.0) as the last resolvable rung before Null.",
        )

    if multiplier is not None:
        try:
            strategy = MultiplierSafetyStockStrategy(multiplier=multiplier)
            return ResolvedSafetyStock(
                strategy=strategy, method=type(strategy).__name__, degraded=True,
                reason="No actuals available to compute forecast error; using a flat multiplier instead.",
            )
        except ValueError:
            pass  # invalid multiplier (< 1.0) -- fall through to Null

    strategy = NullSafetyStockStrategy()
    return ResolvedSafetyStock(
        strategy=strategy, method=type(strategy).__name__, degraded=True,
        reason="No actuals and no usable multiplier given; safety stock defaulted to 0 -- treat with caution.",
    )
