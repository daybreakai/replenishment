from replenishment.strategies.resolver import resolve_safety_stock_strategy
from replenishment.strategies.safety_stock import (
    FillRateSafetyStock, SqrtHorizonSafetyStock, KRmseSafetyStock,
)
from replenishment.strategies.multiplier import (
    MultiplierSafetyStockStrategy, NullSafetyStockStrategy,
)


def test_prefers_fill_rate_when_target_and_actuals_available():
    resolved = resolve_safety_stock_strategy(
        has_actuals=True, periods_observed=10, target_fill_rate=0.95, factor=1.65,
    )
    assert isinstance(resolved.strategy, FillRateSafetyStock)
    assert resolved.method == "FillRateSafetyStock"
    assert resolved.degraded is False


def test_falls_back_to_sqrt_horizon_when_no_fill_rate_target():
    resolved = resolve_safety_stock_strategy(
        has_actuals=True, periods_observed=10, target_fill_rate=None, factor=1.65,
    )
    assert isinstance(resolved.strategy, SqrtHorizonSafetyStock)
    assert resolved.degraded is False


def test_falls_back_to_flat_k_rmse_when_no_factor_given():
    resolved = resolve_safety_stock_strategy(
        has_actuals=True, periods_observed=10, target_fill_rate=None, factor=None,
    )
    assert isinstance(resolved.strategy, KRmseSafetyStock)
    assert resolved.strategy.factor == 1.0
    assert resolved.degraded is True
    assert "flat" in resolved.reason.lower()


def test_falls_back_to_multiplier_when_no_actuals():
    resolved = resolve_safety_stock_strategy(
        has_actuals=False, periods_observed=0, multiplier=1.2,
    )
    assert isinstance(resolved.strategy, MultiplierSafetyStockStrategy)
    assert resolved.degraded is True


def test_falls_back_to_null_when_nothing_available():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    assert isinstance(resolved.strategy, NullSafetyStockStrategy)
    assert resolved.degraded is True
    assert resolved.reason is not None


def test_no_actuals_but_periods_observed_positive_is_not_enough():
    # has_actuals=False must win even if periods_observed is nonzero (defensive:
    # caller passed a stale count without actuals wired in).
    resolved = resolve_safety_stock_strategy(
        has_actuals=False, periods_observed=10, factor=1.65, target_fill_rate=0.95,
    )
    assert isinstance(resolved.strategy, NullSafetyStockStrategy)


def test_falls_back_past_invalid_target_fill_rate_to_sqrt_horizon():
    resolved = resolve_safety_stock_strategy(
        has_actuals=True, periods_observed=10, target_fill_rate=1.5, factor=1.65,
    )
    assert isinstance(resolved.strategy, SqrtHorizonSafetyStock)
    assert resolved.degraded is False


def test_falls_back_to_null_when_multiplier_invalid():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0, multiplier=0.5)
    assert isinstance(resolved.strategy, NullSafetyStockStrategy)
    assert resolved.degraded is True
