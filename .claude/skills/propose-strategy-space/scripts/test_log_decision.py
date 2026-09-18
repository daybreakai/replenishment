"""ponytail self-check for log_decision.py -- run `python test_log_decision.py`."""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from log_decision import log_decision  # noqa: E402

VALID_SPACE = {
    "data_summary": {"demand_classes_present": ["lumpy"]},
    "selection_rationale": "Items are lumpy, so only distributional strategies were tried.",
    "candidates": [
        {"strategy": "CompoundPoissonSafetyStock", "trigger": "OrderUpToTrigger", "param_grid": {"target_service_level": [0.95]}},
    ],
}


def test_valid_space_is_logged_under_customer_dir():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        out_path = log_decision(VALID_SPACE, "Demand is lumpy and we keep stocking out.", "acme", out_dir)
        assert out_path.parent == out_dir / "acme", out_path
        record = yaml.safe_load(out_path.read_text())
        assert record["customer"] == "acme"
        assert record["hypothesis"] == "Demand is lumpy and we keep stocking out."
        assert record["selection_rationale"] == VALID_SPACE["selection_rationale"]
        assert record["candidates"] == VALID_SPACE["candidates"]


def test_invalid_space_is_never_logged():
    invalid_space = {"candidates": [{"strategy": "NotARealStrategy"}]}
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            log_decision(invalid_space, "hypothesis", "acme", out_dir)
        except ValueError as exc:
            assert "invalid" in str(exc)
            assert not any((out_dir / "acme").glob("*")) if (out_dir / "acme").exists() else True
            return
        raise AssertionError("expected ValueError for an invalid strategy space")


def test_two_runs_for_same_customer_produce_two_files():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        p1 = log_decision(VALID_SPACE, "Hypothesis one about lumpy demand.", "acme", out_dir)
        p2 = log_decision(VALID_SPACE, "Hypothesis two about lumpy demand.", "acme", out_dir)
        assert p1 != p2
        assert len(list((out_dir / "acme").glob("*.yaml"))) == 2


def test_path_traversal_customer_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for unsafe in ("../escape", "a/b", "a\\b", "..", "."):
            try:
                log_decision(VALID_SPACE, "hypothesis", unsafe, out_dir)
            except ValueError as exc:
                assert "customer" in str(exc)
                continue
            raise AssertionError(f"expected ValueError for unsafe --customer {unsafe!r}")
        # nothing should have been written outside out_dir
        assert list(out_dir.iterdir()) == []


if __name__ == "__main__":
    test_valid_space_is_logged_under_customer_dir()
    test_invalid_space_is_never_logged()
    test_two_runs_for_same_customer_produce_two_files()
    test_path_traversal_customer_is_rejected()
    print("ok")
