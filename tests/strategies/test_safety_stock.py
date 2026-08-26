import math
import pytest
from replenishment.timeseries import TimeSeries
from replenishment.strategies.safety_stock import (
    SqrtHorizonSafetyStock, KRmseSafetyStock, KMaeSafetyStock,
    FillRateSafetyStock, SafetyStockRangeError,
)


FORECAST = TimeSeries.from_values([10, 10, 10, 10, 10, 10])
ACTUALS = TimeSeries.from_values([10, 12, 8, 11, 9, 10])


def test_sqrt_horizon_scales_error_by_sqrt_of_horizon():
    strategy = SqrtHorizonSafetyStock(factor=1.65)
    ss_h1 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=1, service_level_factor=1.65)
    ss_h4 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=4, service_level_factor=1.65)
    assert abs(ss_h4 - ss_h1 * math.sqrt(4)) < 1e-9


def test_k_rmse_does_not_scale_with_horizon():
    strategy = KRmseSafetyStock(factor=1.65)
    ss_h1 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=1, service_level_factor=1.65)
    ss_h4 = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=4, service_level_factor=1.65)
    assert abs(ss_h1 - ss_h4) < 1e-9


def test_k_mae_uses_mean_absolute_error_not_rmse():
    from replenishment.math_ import mae
    strategy = KMaeSafetyStock(factor=1.0)
    ss = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=1, service_level_factor=1.0)
    expected_mae = mae([10, 12, 8, 11, 9, 10], [10, 10, 10, 10, 10, 10])
    assert abs(ss - expected_mae) < 1e-9


def test_k_rmse_requires_actuals():
    strategy = KRmseSafetyStock(factor=1.65)
    with pytest.raises(ValueError, match="actuals"):
        strategy.compute(forecast=FORECAST, actuals=None, period=6, lead_time=0, horizon=1, service_level_factor=1.65)


def test_safety_stock_zero_at_period_zero_no_history():
    strategy = SqrtHorizonSafetyStock(factor=1.65)
    ss = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=0, lead_time=0, horizon=1, service_level_factor=1.65)
    assert ss == 0.0


def test_fill_rate_computes_positive_safety_stock_in_range():
    strategy = FillRateSafetyStock(target_fill_rate=0.95)
    ss = strategy.compute(forecast=FORECAST, actuals=ACTUALS, period=6, lead_time=0, horizon=1, service_level_factor=0.95)
    assert ss > 0


def test_fill_rate_zero_std_dev_returns_zero():
    flat_forecast = TimeSeries.from_values([10, 10, 10])
    flat_actuals = TimeSeries.from_values([10, 10, 10])
    strategy = FillRateSafetyStock(target_fill_rate=0.95)
    ss = strategy.compute(forecast=flat_forecast, actuals=flat_actuals, period=3, lead_time=0, horizon=1, service_level_factor=0.95)
    assert ss == 0.0


def test_fill_rate_raises_by_default_when_out_of_solvable_range():
    # An essentially-impossible fill rate for this error distribution:
    # loss_target computed will be so small inverse_normal_loss would
    # need z beyond the +6sigma upper bound to satisfy it exactly.
    # Verified via scratch script (see task-4-report.md): with FORECAST
    # ([10]*6, extends to mean_demand=10) and this spike, loss_target
    # ~= 1.44e-11 while normal_loss(upper_bound=6) ~= 1.56e-10, so
    # loss_target < normal_loss(upper_bound) with a comfortable margin
    # (an earlier, smaller spike of 500 landed within ~6% of the
    # boundary -- too fragile to rely on).
    tiny_error_actuals = TimeSeries.from_values([10] * 50 + [10, 5000])  # one huge miss
    strategy = FillRateSafetyStock(target_fill_rate=0.999999999)
    with pytest.raises(SafetyStockRangeError):
        strategy.compute(forecast=FORECAST, actuals=tiny_error_actuals, period=52, lead_time=0, horizon=1, service_level_factor=0.999999999)


def test_fill_rate_clip_mode_returns_boundary_instead_of_raising():
    tiny_error_actuals = TimeSeries.from_values([10] * 50 + [10, 5000])
    strategy = FillRateSafetyStock(target_fill_rate=0.999999999, on_out_of_range="clip")
    ss = strategy.compute(forecast=FORECAST, actuals=tiny_error_actuals, period=52, lead_time=0, horizon=1, service_level_factor=0.999999999)
    assert ss > 0  # did not raise; returned the clipped boundary value


def test_fill_rate_rejects_target_outside_open_interval():
    with pytest.raises(ValueError):
        FillRateSafetyStock(target_fill_rate=1.0)


def test_fixed_error_flat_is_factor_times_error_regardless_of_history():
    from replenishment.strategies.safety_stock import FixedErrorSafetyStock
    strategy = FixedErrorSafetyStock(error=4.0, factor=1.5)
    # no actuals needed, period 0 fine -- the whole point is pre-window error
    ss = strategy.compute(forecast=FORECAST, actuals=None, period=0, lead_time=1, horizon=1, service_level_factor=1.5)
    assert ss == 6.0


def test_fixed_error_scaled_matches_sqrt_horizon_shape():
    from replenishment.strategies.safety_stock import FixedErrorSafetyStock
    strategy = FixedErrorSafetyStock(error=4.0, factor=1.5, scale_by_horizon=True)
    ss = strategy.compute(forecast=FORECAST, actuals=None, period=0, lead_time=1, horizon=3, service_level_factor=1.5)
    assert abs(ss - 1.5 * 4.0 * math.sqrt(4)) < 1e-9


def test_fixed_error_rejects_negative_error():
    from replenishment.strategies.safety_stock import FixedErrorSafetyStock
    with pytest.raises(ValueError):
        FixedErrorSafetyStock(error=-1.0)
