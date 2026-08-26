"""Demand-pattern classification.

Ported from supplyplanning/prototype/engine (Syntetos-Boylan, 2005). Not a
SafetyStockStrategy itself -- a router input other strategies (or the
caller) can use to pick among KRmseSafetyStock / KingsFormulaSafetyStock /
CompoundPoissonSafetyStock for a given series.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Literal

DemandClass = Literal["smooth", "erratic", "intermittent", "lumpy"]
ABCTier = Literal["A", "B", "C"]
XYZTier = Literal["X", "Y", "Z"]

_ADI_THRESHOLD = 1.32
_CV2_THRESHOLD = 0.49


def classify_demand(history: list[float]) -> DemandClass:
    """Classify a demand series per Syntetos-Boylan (2005).

    Boundaries: ADI (avg inter-demand interval) = 1.32, CV^2 = 0.49.

    Args:
        history: per-period demand, zeros allowed.

    Returns:
        One of {smooth, erratic, intermittent, lumpy}.
    """
    if not history:
        return "smooth"

    nonzero_indices = [i for i, v in enumerate(history) if v > 0]
    if not nonzero_indices:
        return "intermittent"  # all zeros

    if len(nonzero_indices) <= 1:
        adi = float(len(history))  # single-event series -> very intermittent
    else:
        gaps = [b - a for a, b in zip(nonzero_indices, nonzero_indices[1:])]
        adi = statistics.fmean(gaps)

    nonzero = [history[i] for i in nonzero_indices]
    if len(nonzero) > 1 and statistics.fmean(nonzero) > 0:
        cv2 = (statistics.stdev(nonzero) / statistics.fmean(nonzero)) ** 2
    else:
        cv2 = 0.0

    if adi < _ADI_THRESHOLD and cv2 < _CV2_THRESHOLD:
        return "smooth"
    if adi < _ADI_THRESHOLD and cv2 >= _CV2_THRESHOLD:
        return "erratic"
    if adi >= _ADI_THRESHOLD and cv2 < _CV2_THRESHOLD:
        return "intermittent"
    return "lumpy"


def classify_abc(
    item_values: Mapping[str, float], cutoffs: tuple[float, float] = (0.8, 0.95)
) -> dict[str, ABCTier]:
    """Rank items by value (e.g. annual $ usage) and bucket by cumulative
    share: A = top items making up cutoffs[0] of total value, B = up to
    cutoffs[1], C = the long tail. Classic Pareto/ABC analysis.

    Args:
        item_values: unique_id -> value (must be >= 0).
        cutoffs: (a_cutoff, b_cutoff), 0 < a_cutoff < b_cutoff <= 1.

    Returns:
        unique_id -> "A" | "B" | "C". All items are "C" when total value is 0.
    """
    a_cutoff, b_cutoff = cutoffs
    if not (0 < a_cutoff < b_cutoff <= 1):
        raise ValueError(f"cutoffs must satisfy 0 < a_cutoff < b_cutoff <= 1, got {cutoffs!r}.")
    if any(v < 0 for v in item_values.values()):
        raise ValueError("item_values must be non-negative.")

    total = sum(item_values.values())
    if total == 0:
        return {uid: "C" for uid in item_values}

    ranked = sorted(item_values.items(), key=lambda kv: kv[1], reverse=True)
    result: dict[str, ABCTier] = {}
    running = 0.0
    for uid, value in ranked:
        running += value
        share = running / total
        if share <= a_cutoff:
            result[uid] = "A"
        elif share <= b_cutoff:
            result[uid] = "B"
        else:
            result[uid] = "C"
    return result


def classify_xyz(cv: float, thresholds: tuple[float, float] = (0.5, 1.0)) -> XYZTier:
    """Bucket a single item's demand coefficient of variation into a
    variability tier: X = steady, Y = variable, Z = erratic.

    Args:
        cv: coefficient of variation (stdev / mean) of period demand, >= 0.
        thresholds: (x_max, y_max), 0 < x_max < y_max.

    Returns:
        "X" if cv < x_max, "Y" if cv < y_max, else "Z".
    """
    x_max, y_max = thresholds
    if not (0 < x_max < y_max):
        raise ValueError(f"thresholds must satisfy 0 < x_max < y_max, got {thresholds!r}.")
    if cv < 0:
        raise ValueError(f"cv must be non-negative, got {cv!r}.")
    if cv < x_max:
        return "X"
    if cv < y_max:
        return "Y"
    return "Z"
