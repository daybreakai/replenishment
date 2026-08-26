"""Demand-pattern classification.

Ported from supplyplanning/prototype/engine (Syntetos-Boylan, 2005). Not a
SafetyStockStrategy itself -- a router input other strategies (or the
caller) can use to pick among KRmseSafetyStock / KingsFormulaSafetyStock /
CompoundPoissonSafetyStock for a given series.
"""
from __future__ import annotations

import statistics
from typing import Literal

DemandClass = Literal["smooth", "erratic", "intermittent", "lumpy"]

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
