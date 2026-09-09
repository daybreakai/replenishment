"""ponytail self-check for backtest.py -- run `python test_backtest.py`."""
import csv
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from backtest import _aggregate, _log_results, run_backtest, run_sweep  # noqa: E402


def _base_args(**overrides):
    base = dict(
        strategy=None, params=None, trigger=None, configs=None,
        no_baseline=False, log_path=None, no_log=True,
        data="synthetic", n_items=3, periods=30, seed=7,
        lead_time=None, lead_time_field=None,
        review_period=None, review_period_field=None,
        moq=None, moq_field=None,
        horizon=1, forecast_field=None,
        holding_cost=None, stockout_cost=None, order_cost=None,
        actuals_field=None, initial_on_hand_field=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_synthetic_backtest_runs_for_every_item():
    args = _base_args(strategy="KRmseSafetyStock", params="factor=1.65", trigger="OrderUpToTrigger")
    outcome = run_backtest(args)
    assert len(outcome["results"]) == 3, outcome
    assert not outcome["errors"], outcome["errors"]


def test_cost_breakdown_and_turnover_are_consistent():
    args = _base_args(strategy="KRmseSafetyStock", params="factor=1.65", trigger="OrderUpToTrigger")
    outcome = run_backtest(args)
    agg = _aggregate(outcome)
    assert abs((agg["holding_cost"] + agg["stockout_cost"] + agg["ordering_cost"]) - agg["total_cost"]) < 1e-6, agg
    assert agg["turnover"] > 0, agg
    assert agg["total_orders"] >= 0, agg


def test_unknown_strategy_param_is_rejected():
    args = _base_args(strategy="KRmseSafetyStock", params="not_a_param=1", trigger="OrderUpToTrigger", n_items=1, periods=10)
    try:
        run_backtest(args)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown strategy param")


def _sweep_args(**overrides):
    base = dict(
        configs='[{"strategy": "KRmseSafetyStock", "params": {"factor": 1.65}}, '
                '{"strategy": "SqrtHorizonSafetyStock", "params": {"factor": 1.65}}]',
    )
    base.update(overrides)
    return _base_args(**base)


def test_sweep_inserts_one_baseline_row_per_trigger():
    outcomes = run_sweep(_sweep_args())
    assert len(outcomes) == 3, outcomes
    baselines = [o for o in outcomes if o["is_baseline"]]
    assert len(baselines) == 1 and baselines[0]["strategy"] == "NullSafetyStockStrategy", outcomes


def test_no_baseline_flag_skips_baseline_row():
    outcomes = run_sweep(_sweep_args(no_baseline=True))
    assert len(outcomes) == 2, outcomes
    assert not any(o["is_baseline"] for o in outcomes)


def test_log_results_appends_one_line():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "results.jsonl"
        args = _sweep_args(log_path=str(log_path), no_log=False)
        outcomes = run_sweep(args)
        _log_results(outcomes, args)
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1, lines
        record = json.loads(lines[0])
        assert record["data_source"] == "synthetic"
        assert len(record["rows"]) == 3, record
        assert record["rows"][0]["forecast_field"] == "forecast", record


def _write_moq_csv(path: Path) -> None:
    fieldnames = [
        "unique_id", "ds", "demand", "forecast", "actuals", "holding_cost_per_unit",
        "stockout_cost_per_unit", "order_cost_per_order", "lead_time", "initial_on_hand",
        "current_stock", "is_forecast", "moq", "review_period",
    ]
    rows = []
    per_item = {"A": {"moq": 6, "review_period": 2}, "B": {"moq": 12, "review_period": 4}}
    for uid, cfg in per_item.items():
        for period in range(10):
            rows.append({
                "unique_id": uid, "ds": f"2024-01-{period + 1:02d}", "demand": 10, "forecast": 10,
                "actuals": 10, "holding_cost_per_unit": 1.0, "stockout_cost_per_unit": 5.0,
                "order_cost_per_order": 0.0, "lead_time": 2, "initial_on_hand": 20,
                "current_stock": 20, "is_forecast": False,
                "moq": cfg["moq"], "review_period": cfg["review_period"],
            })
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_moq_field_and_review_period_field_are_per_item():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "moq.csv"
        _write_moq_csv(csv_path)
        args = _base_args(
            strategy="NullSafetyStockStrategy", params=None, trigger="OrderUpToTrigger",
            data=str(csv_path), moq_field="moq", review_period_field="review_period",
        )
        outcome = run_backtest(args)
        assert not outcome["errors"], outcome["errors"]
        assert len(outcome["results"]) == 2, outcome


def test_moq_field_rejects_synthetic_data():
    args = _base_args(strategy="NullSafetyStockStrategy", trigger="OrderUpToTrigger", moq_field="moq")
    try:
        run_backtest(args)
    except ValueError as exc:
        assert "synthetic" in str(exc)
        return
    raise AssertionError("expected ValueError for --moq-field with synthetic data")


def test_moq_field_missing_column_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "moq.csv"
        _write_moq_csv(csv_path)
        args = _base_args(
            strategy="NullSafetyStockStrategy", trigger="OrderUpToTrigger",
            data=str(csv_path), moq_field="not_a_real_column",
        )
        try:
            run_backtest(args)
        except ValueError as exc:
            assert "not_a_real_column" in str(exc)
            return
        raise AssertionError("expected ValueError for a missing --moq-field column")


def test_moq_and_moq_field_are_mutually_exclusive():
    args = _base_args(strategy="NullSafetyStockStrategy", trigger="OrderUpToTrigger", moq=6, moq_field="moq")
    try:
        run_backtest(args)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
        return
    raise AssertionError("expected ValueError for --moq + --moq-field together")


def test_forecast_field_reads_percentile_column():
    args = _sweep_args(configs=json.dumps([
        {"strategy": "NullSafetyStockStrategy", "forecast_field": "p90"},
        {"strategy": "KRmseSafetyStock", "params": {"factor": 1.65}},
    ]))
    outcomes = run_sweep(args)
    variants = [o for o in outcomes if not o["is_baseline"]]
    by_strategy = {(o["strategy"], o["forecast_field"]) for o in variants}
    assert ("NullSafetyStockStrategy", "p90") in by_strategy, variants
    assert ("KRmseSafetyStock", "forecast") in by_strategy, variants
    assert not any(o["errors"] for o in variants), variants


def test_unknown_forecast_field_is_rejected():
    args = _sweep_args(configs=json.dumps([{"strategy": "NullSafetyStockStrategy", "forecast_field": "p999"}]),
                        no_baseline=True)
    try:
        run_sweep(args)
    except ValueError as exc:
        assert "p999" in str(exc)
        return
    raise AssertionError("expected ValueError for an unknown forecast_field")


if __name__ == "__main__":
    test_synthetic_backtest_runs_for_every_item()
    test_cost_breakdown_and_turnover_are_consistent()
    test_unknown_strategy_param_is_rejected()
    test_sweep_inserts_one_baseline_row_per_trigger()
    test_no_baseline_flag_skips_baseline_row()
    test_log_results_appends_one_line()
    test_moq_field_and_review_period_field_are_per_item()
    test_moq_field_rejects_synthetic_data()
    test_moq_field_missing_column_is_reported()
    test_moq_and_moq_field_are_mutually_exclusive()
    test_forecast_field_reads_percentile_column()
    test_unknown_forecast_field_is_rejected()
    print("ok")
