"""Flat registry of segmentation strategies plus their pure calculation
logic. Source of truth for which extra tables a strategy needs -- see
../references/extra-data-requirements.md for the human-readable rationale
and ../SKILL.md for the workflow.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from replenishment.classification import classify_demand  # noqa: E402

SEGMENTATION_STRATEGIES = {
    "adi_cv2": {
        "required_tables": ["t_outbound_shipment"],
        "description": (
            "Demand-timing/variability class (smooth/erratic/intermittent/"
            "lumpy) via Syntetos-Boylan ADI/CV2 on raw shipped_qty history."
        ),
    },
    "abc_revenue": {
        "required_tables": ["t_outbound_shipment", "t_product"],
        "description": (
            "Revenue = shipped_qty * unit_cost, Pareto-bucketed into A (top "
            "80% cumulative revenue) / B (next 15%) / C (last 5%)."
        ),
    },
    "xyz_variability": {
        "required_tables": ["t_outbound_shipment", "t_product"],
        "description": (
            "Coefficient of variation of per-period revenue, bucketed "
            "X (<=0.5) / Y (0.5-1.0) / Z (>1.0)."
        ),
    },
    "abc_xyz_matrix": {
        "required_tables": ["t_outbound_shipment", "t_product"],
        "description": (
            "3x3 cross of abc_revenue and xyz_variability (e.g. 'AX', "
            "'CZ'). Requires both to have already been computed."
        ),
    },
    "abc_revenue_by_hierarchy": {
        "required_tables": ["t_outbound_shipment", "t_product", "t_product_hierarchy"],
        "description": (
            "Same as abc_revenue, but revenue is rolled up by "
            "t_product.product_group_id (via t_product_hierarchy) before "
            "bucketing -- segments a product group, not a single SKU."
        ),
    },
}


def adi_cv2_class(history: list[float]) -> str:
    return classify_demand(history)


def abc_bucket(revenue_by_id: dict[str, float]) -> dict[str, str]:
    """Pareto 80/15/5 cumulative-revenue buckets -> A/B/C."""
    ranked = sorted(revenue_by_id.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in ranked)
    out = {}
    running = 0.0
    for uid, rev in ranked:
        running += rev
        share = running / total if total else 1.0
        out[uid] = "A" if share <= 0.80 else ("B" if share <= 0.95 else "C")
    return out


def xyz_bucket(revenue_series_by_id: dict[str, list[float]]) -> dict[str, str]:
    """CV of per-period revenue -> X (<=0.5) / Y (0.5-1.0) / Z (>1.0)."""
    out = {}
    for uid, series in revenue_series_by_id.items():
        nonzero = [v for v in series if v]
        if len(nonzero) < 2 or statistics.fmean(nonzero) == 0:
            out[uid] = "Z"  # too little signal to call it stable
            continue
        cv = statistics.stdev(nonzero) / statistics.fmean(nonzero)
        out[uid] = "X" if cv <= 0.5 else ("Y" if cv <= 1.0 else "Z")
    return out


def abc_xyz_matrix(abc_by_id: dict[str, str], xyz_by_id: dict[str, str]) -> dict[str, str]:
    return {uid: f"{abc_by_id[uid]}{xyz_by_id[uid]}" for uid in abc_by_id if uid in xyz_by_id}
