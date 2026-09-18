"""ponytail self-check for validate_segmentation.py -- run `python test_validate_segmentation.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_segmentation import validate  # noqa: E402


def test_valid_request_has_no_errors():
    assert validate(["adi_cv2"], ["t_outbound_shipment"]) == []


def test_missing_required_table_is_a_hard_error():
    errors = validate(["abc_revenue"], ["t_outbound_shipment"])
    assert errors and "t_product" in errors[0], errors


def test_unknown_strategy_is_rejected():
    errors = validate(["not_a_real_strategy"], ["t_outbound_shipment"])
    assert errors and "unknown strategy" in errors[0], errors


def test_hierarchy_strategy_requires_hierarchy_table():
    errors = validate(["abc_revenue_by_hierarchy"], ["t_outbound_shipment", "t_product"])
    assert errors and "t_product_hierarchy" in errors[0], errors
    ok = validate(["abc_revenue_by_hierarchy"], ["t_outbound_shipment", "t_product", "t_product_hierarchy"])
    assert ok == [], ok


if __name__ == "__main__":
    test_valid_request_has_no_errors()
    test_missing_required_table_is_a_hard_error()
    test_unknown_strategy_is_rejected()
    test_hierarchy_strategy_requires_hierarchy_table()
    print("ok")
