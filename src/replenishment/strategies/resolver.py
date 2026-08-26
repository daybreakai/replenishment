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
    """The strategy resolve_safety_stock_strategy picked, plus how it got there.

    strategy: the constructed SafetyStockStrategy instance to use.
    method: strategy's class name, for logging/reporting without needing
        to introspect the instance.
    degraded: True if this is not the highest-fidelity rung the ladder
        offers (either because better inputs weren't given, or because
        a given input was invalid and got skipped past).
    reason: human-readable explanation when degraded=True (or when a
        supplied-but-unusable input was silently skipped even on a
        non-degraded rung); None when the top rung resolved cleanly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

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
    """Pick a SafetyStockStrategy from what data is actually available,
    degrading gracefully instead of raising. Rungs, highest fidelity first:

    1. has_actuals + periods_observed > 0 + valid target_fill_rate
       -> FillRateSafetyStock (exact service-level semantics).
    2. has_actuals + periods_observed > 0 + factor
       -> SqrtHorizonSafetyStock (scales with lead time + horizon).
    3. has_actuals + periods_observed > 0, neither of the above resolves
       -> KRmseSafetyStock(factor=1.0), flat fallback, degraded=True.
    4. no actuals + valid multiplier -> MultiplierSafetyStockStrategy, degraded=True.
    5. nothing usable -> NullSafetyStockStrategy, degraded=True.

    An invalid target_fill_rate or multiplier (rejected by the strategy's
    own validation) is treated the same as not having been given, except
    the resulting ResolvedSafetyStock is always marked degraded=True with
    a reason naming what was rejected -- even on a rung that would
    otherwise report degraded=False -- so a caller's typo never produces a
    falsely-confident result. Never raises.
    """
    if has_actuals and periods_observed > 0:
        fill_rate_rejected_reason: str | None = None
        if target_fill_rate is not None:
            try:
                strategy = FillRateSafetyStock(target_fill_rate=target_fill_rate)
            except ValueError as exc:
                fill_rate_rejected_reason = (
                    f"target_fill_rate={target_fill_rate!r} was invalid ({exc}); fell back past it."
                )
            else:
                return ResolvedSafetyStock(strategy=strategy, method=type(strategy).__name__, degraded=False)
        if factor is not None:
            strategy = SqrtHorizonSafetyStock(factor=factor)
            reason = fill_rate_rejected_reason
            if multiplier is not None:
                extra = f"multiplier={multiplier!r} was also given but is unused on this rung."
                reason = f"{reason} {extra}" if reason else extra
            return ResolvedSafetyStock(
                strategy=strategy, method=type(strategy).__name__,
                degraded=reason is not None,
                reason=reason,
            )
        strategy = KRmseSafetyStock(factor=1.0)
        reason = ("No target_fill_rate or factor given; using a flat K-RMSE buffer (factor=1.0) "
                  "as the last resolvable rung before Null.")
        if fill_rate_rejected_reason is not None:
            reason = fill_rate_rejected_reason + " No factor given either; using a flat K-RMSE buffer (factor=1.0)."
        if multiplier is not None:
            reason += f" multiplier={multiplier!r} was also given but is unused on this rung."
        return ResolvedSafetyStock(strategy=strategy, method=type(strategy).__name__, degraded=True, reason=reason)

    if multiplier is not None:
        try:
            strategy = MultiplierSafetyStockStrategy(multiplier=multiplier)
        except ValueError as exc:
            strategy = NullSafetyStockStrategy()
            return ResolvedSafetyStock(
                strategy=strategy, method=type(strategy).__name__, degraded=True,
                reason=f"multiplier={multiplier!r} was invalid ({exc}); no actuals available either; "
                       "safety stock defaulted to 0 -- treat with caution.",
            )
        return ResolvedSafetyStock(
            strategy=strategy, method=type(strategy).__name__, degraded=True,
            reason="No actuals available to compute forecast error; using a flat multiplier instead.",
        )

    strategy = NullSafetyStockStrategy()
    return ResolvedSafetyStock(
        strategy=strategy, method=type(strategy).__name__, degraded=True,
        reason="No actuals and no usable multiplier given; safety stock defaulted to 0 -- treat with caution.",
    )
