"""ponytail self-check for validate_strategy_space.py -- run
`python test_validate_strategy_space.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_strategy_space import validate  # noqa: E402


def _valid_space(**overrides):
    space = {
        "data_summary": {"demand_classes_present": ["smooth"]},
        "selection_rationale": "Items classified smooth; KRmse fits regular low-variance demand directly.",
        "candidates": [
            {"strategy": "KRmseSafetyStock", "trigger": "OrderUpToTrigger", "param_grid": {"factor": [1.0, 1.65]}},
        ],
    }
    space.update(overrides)
    return space


def test_valid_space_has_no_errors():
    assert validate(_valid_space()) == []


def test_unknown_strategy_is_reported():
    space = {"candidates": [{"strategy": "NotARealStrategy"}]}
    errors = validate(space)
    assert any("NotARealStrategy" in e for e in errors), errors


def test_unknown_trigger_is_reported():
    space = {"candidates": [{"strategy": "NullSafetyStockStrategy", "trigger": "NotARealTrigger"}]}
    errors = validate(space)
    assert any("NotARealTrigger" in e for e in errors), errors


def test_missing_required_param_is_reported():
    space = {"candidates": [{"strategy": "FillRateSafetyStock", "param_grid": {}}]}
    errors = validate(space)
    assert any("target_fill_rate" in e for e in errors), errors


def test_unknown_param_is_reported():
    space = {"candidates": [{"strategy": "KRmseSafetyStock", "param_grid": {"not_a_param": [1]}}]}
    errors = validate(space)
    assert any("not_a_param" in e for e in errors), errors


def test_non_list_param_grid_value_is_reported():
    space = {"candidates": [{"strategy": "KRmseSafetyStock", "param_grid": {"factor": 1.65}}]}
    errors = validate(space)
    assert any("must be a non-empty list" in e for e in errors), errors


def test_empty_candidates_is_reported():
    assert validate({"candidates": []})
    assert validate({})


def test_too_many_candidates_is_a_hard_rule():
    # 11 safety-stock strategies today -> cap is 5; 6 distinct real strategies should fail.
    six_strategies = [
        "KRmseSafetyStock", "SqrtHorizonSafetyStock", "KMaeSafetyStock", "FixedErrorSafetyStock",
        "MultiplierSafetyStockStrategy", "NullSafetyStockStrategy",
    ]
    space = _valid_space(candidates=[{"strategy": s} for s in six_strategies])
    errors = validate(space)
    assert any("at most" in e for e in errors), errors


def test_missing_selection_rationale_is_reported():
    space = _valid_space()
    del space["selection_rationale"]
    errors = validate(space)
    assert any("selection_rationale" in e for e in errors), errors


def test_ungrounded_selection_rationale_is_reported():
    space = _valid_space(selection_rationale="We tried a few options that seemed reasonable.")
    errors = validate(space)
    assert any("doesn't reference any of" in e for e in errors), errors


def test_object_typed_param_strategy_is_rejected():
    # DemandBufferDecorator.wrapped needs a strategy *instance*, which a
    # JSON param_grid can't express -- must be rejected outright, not
    # allowed through to fail obscurely at simulation time.
    space = _valid_space(candidates=[
        {"strategy": "DemandBufferDecorator", "param_grid": {"wrapped": ["KRmseSafetyStock"], "strength": [0.2]}},
    ])
    errors = validate(space)
    assert any("wrapped" in e and "strategy-object" in e for e in errors), errors


def test_grounded_selection_rationale_passes():
    space = _valid_space(
        data_summary={"demand_classes_present": ["lumpy"]},
        selection_rationale="Items are lumpy, so only distributional strategies were tried.",
        candidates=[{"strategy": "CompoundPoissonSafetyStock", "param_grid": {"target_service_level": [0.95]}}],
    )
    assert validate(space) == []


if __name__ == "__main__":
    test_valid_space_has_no_errors()
    test_unknown_strategy_is_reported()
    test_unknown_trigger_is_reported()
    test_missing_required_param_is_reported()
    test_unknown_param_is_reported()
    test_non_list_param_grid_value_is_reported()
    test_empty_candidates_is_reported()
    test_too_many_candidates_is_a_hard_rule()
    test_missing_selection_rationale_is_reported()
    test_ungrounded_selection_rationale_is_reported()
    test_object_typed_param_strategy_is_rejected()
    test_grounded_selection_rationale_passes()
    print("ok")
