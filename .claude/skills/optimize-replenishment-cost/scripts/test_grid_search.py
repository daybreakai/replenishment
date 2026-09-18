"""ponytail self-check for grid_search.py -- run `python test_grid_search.py`."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run-replenishment-backtest" / "scripts"))
import backtest  # noqa: E402
import grid_search as grid_search_mod  # noqa: E402
from grid_search import _build_arg_parser, _DEAD_BACKTEST_FLAGS, _print_errors, expand_candidates, run_grid_search  # noqa: E402


def _base_args(**overrides):
    base = dict(
        strategy_space=None, min_fill_rate=None, max_total_cost=None, top=5,
        data="synthetic", n_items=3, periods=30, seed=7,
        lead_time=None, lead_time_field=None,
        review_period=None, review_period_field=None,
        moq=None, moq_field=None,
        horizon=1, holding_cost=None, stockout_cost=None, order_cost=None,
        actuals_field=None, initial_on_hand_field=None,
        strategy=None, params=None, trigger=None, configs=None,
        no_baseline=True, log_path=None, no_log=True, forecast_field=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_space(tmp: Path, candidates: list[dict]) -> Path:
    path = tmp / "space.json"
    path.write_text(json.dumps({
        "hypothesis": "test",
        "data_summary": {"demand_classes_present": ["smooth"]},
        "selection_rationale": "Items classified smooth; testing against that.",
        "candidates": candidates,
    }))
    return path


def test_expand_candidates_takes_cartesian_product_of_param_grid():
    space = {"candidates": [{"strategy": "KRmseSafetyStock", "param_grid": {"factor": [1.0, 1.5, 2.0]}}]}
    configs = expand_candidates(space)
    assert len(configs) == 3, configs
    assert {c["params"]["factor"] for c in configs} == {1.0, 1.5, 2.0}


def test_expand_candidates_with_no_grid_yields_one_config():
    space = {"candidates": [{"strategy": "NullSafetyStockStrategy"}]}
    configs = expand_candidates(space)
    assert configs == [
        {"strategy": "NullSafetyStockStrategy", "params": {}, "trigger": "OrderUpToTrigger", "forecast_field": "forecast"},
    ], configs


def test_multi_param_grid_is_full_cartesian_product():
    space = {"candidates": [{"strategy": "DemandBufferDecorator", "param_grid": {"strength": [0.1, 0.5]}}]}
    configs = expand_candidates(space)
    assert len(configs) == 2, configs


def test_grid_search_ranks_survivors_by_ascending_cost():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        candidates = [{"strategy": "KRmseSafetyStock", "param_grid": {"factor": [0.0, 1.65, 5.0]}}]
        space_path = _write_space(tmp_path, candidates)
        args = _base_args(strategy_space=str(space_path), min_fill_rate=0.0)
        result = run_grid_search(args)
        assert result["n_configs"] == 3
        assert len(result["survivors"]) == 3, result
        costs = [backtest._aggregate(o)["total_cost"] for o in result["survivors"]]
        assert costs == sorted(costs), costs


def test_unsatisfiable_constraint_yields_no_survivors():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        candidates = [{"strategy": "NullSafetyStockStrategy"}]
        space_path = _write_space(tmp_path, candidates)
        args = _base_args(strategy_space=str(space_path), min_fill_rate=0.9999999)
        result = run_grid_search(args)
        assert result["survivors"] == [], result


def test_invalid_strategy_space_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        candidates = [{"strategy": "NotARealStrategy"}]
        space_path = _write_space(tmp_path, candidates)
        args = _base_args(strategy_space=str(space_path))
        try:
            run_grid_search(args)
        except ValueError as exc:
            assert "NotARealStrategy" in str(exc)
            return
        raise AssertionError("expected ValueError for an unknown strategy in the space")


def test_dead_backtest_flags_are_not_in_help():
    parser = _build_arg_parser()
    remaining_dests = {a.dest for a in parser._actions}
    assert not (remaining_dests & _DEAD_BACKTEST_FLAGS), remaining_dests & _DEAD_BACKTEST_FLAGS
    for group in parser._action_groups:
        group_dests = {a.dest for a in group._group_actions}
        assert not (group_dests & _DEAD_BACKTEST_FLAGS), group_dests & _DEAD_BACKTEST_FLAGS
    # sanity check the real flags are still there
    assert "strategy_space" in remaining_dests
    assert "data" in remaining_dests


def test_errored_config_is_surfaced_in_report():
    outcome_with_errors = {
        "strategy": "KRmseSafetyStock", "params": {"factor": 1.65}, "trigger": "OrderUpToTrigger",
        "errors": [("item-1", "boom")],
    }
    outcome_clean = {
        "strategy": "NullSafetyStockStrategy", "params": {}, "trigger": "OrderUpToTrigger", "errors": [],
    }
    result = {"outcomes": [outcome_with_errors, outcome_clean]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_errors(result)
    output = buf.getvalue()
    assert "1 config(s) had item-level errors" in output, output
    assert "KRmseSafetyStock" in output, output
    assert "NullSafetyStockStrategy" not in output, output


def test_no_errors_prints_nothing():
    result = {"outcomes": [{"strategy": "NullSafetyStockStrategy", "params": {}, "trigger": "OrderUpToTrigger", "errors": []}]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_errors(result)
    assert buf.getvalue() == ""


def test_empty_candidates_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        space_path = _write_space(tmp_path, [])
        args = _base_args(strategy_space=str(space_path))
        try:
            run_grid_search(args)
        except ValueError:
            return
        raise AssertionError("expected ValueError for an empty strategy space")


def test_main_rejects_bad_customer_before_running_grid_search(monkeypatch):
    def fail_run_grid_search(args):
        raise AssertionError("run_grid_search must not be called when --customer fails validation")
    monkeypatch.setattr(grid_search_mod, "run_grid_search", fail_run_grid_search)

    original_argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        space_path = _write_space(Path(tmp), [{"strategy": "KRmseSafetyStock", "param_grid": {"factor": [1.65]}}])
        sys.argv = [
            "grid_search.py", "--data", "synthetic", "--strategy-space", str(space_path),
            "--customer", "../evil", "--no-log",
        ]
        try:
            try:
                grid_search_mod.main()
            except SystemExit as exc:
                assert exc.code == 1
                return
            raise AssertionError("expected SystemExit for a bad --customer")
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_expand_candidates_takes_cartesian_product_of_param_grid()
    test_expand_candidates_with_no_grid_yields_one_config()
    test_multi_param_grid_is_full_cartesian_product()
    test_grid_search_ranks_survivors_by_ascending_cost()
    test_unsatisfiable_constraint_yields_no_survivors()
    test_invalid_strategy_space_is_rejected()
    test_dead_backtest_flags_are_not_in_help()
    test_errored_config_is_surfaced_in_report()
    test_no_errors_prints_nothing()
    test_empty_candidates_is_rejected()
    test_main_rejects_bad_customer_before_running_grid_search(_FakeMonkeypatch())
    print("ok")
