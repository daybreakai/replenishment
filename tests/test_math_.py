import math
import pytest
from replenishment.math_ import (
    normal_quantile, normal_pdf, normal_cdf, normal_loss,
    inverse_normal_loss, rmse, mae,
)


def test_normal_quantile_median_is_zero():
    assert abs(normal_quantile(0.5)) < 1e-6


def test_normal_quantile_matches_known_97_5_percentile():
    # standard normal 97.5th percentile is ~1.959964
    assert abs(normal_quantile(0.975) - 1.959964) < 1e-4


def test_normal_quantile_rejects_out_of_range():
    with pytest.raises(ValueError):
        normal_quantile(0.0)
    with pytest.raises(ValueError):
        normal_quantile(1.0)


def test_normal_pdf_peak_at_zero():
    assert abs(normal_pdf(0.0) - 1 / math.sqrt(2 * math.pi)) < 1e-9


def test_normal_cdf_median_is_half():
    assert abs(normal_cdf(0.0) - 0.5) < 1e-9


def test_inverse_normal_loss_roundtrips_within_range():
    for z in (-2.0, -0.5, 0.0, 1.0, 2.0):
        loss = normal_loss(z)
        assert abs(inverse_normal_loss(loss) - z) < 1e-3


def test_rmse_basic():
    assert abs(rmse([10, 12, 8], [10, 10, 10]) - math.sqrt((0 + 4 + 4) / 3)) < 1e-9


def test_mae_basic():
    assert abs(mae([10, 12, 8], [10, 10, 10]) - (0 + 2 + 2) / 3) < 1e-9


def test_rmse_empty_series_is_zero():
    assert rmse([], []) == 0.0
