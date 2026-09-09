"""ponytail self-check for history.py -- run `python test_history.py`."""
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from history import diff_strategies, print_history  # noqa: E402


def _write_two_strategy_log(log_path: Path) -> None:
    log_path.write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:00+00:00", "data_source": "synthetic",
        "rows": [
            {"is_baseline": False, "strategy": "KRmseSafetyStock", "params": {"factor": 1.65},
             "trigger": "OrderUpToTrigger", "n_items": 3, "n_errors": 0,
             "mean_fill_rate": 0.95, "median_fill_rate": 0.95, "total_cost": 100.0},
            {"is_baseline": False, "strategy": "SqrtHorizonSafetyStock", "params": {},
             "trigger": "OrderUpToTrigger", "n_items": 3, "n_errors": 0,
             "mean_fill_rate": 0.90, "median_fill_rate": 0.90, "total_cost": 200.0},
        ],
    }) + "\n")


def test_missing_log_prints_friendly_message():
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_history(Path("/tmp/definitely-does-not-exist-results.jsonl"))
    assert "no runs logged yet" in buf.getvalue().lower()


def test_filters_by_strategy():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "results.jsonl"
        _write_two_strategy_log(log_path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_history(log_path, strategy="KRmseSafetyStock")
        output = buf.getvalue()
        assert "KRmseSafetyStock" in output and "SqrtHorizonSafetyStock" not in output, output


def test_diff_shows_both_strategies_and_correct_delta():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "results.jsonl"
        _write_two_strategy_log(log_path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            diff_strategies(log_path, "KRmseSafetyStock", "SqrtHorizonSafetyStock")
        output = buf.getvalue()
        assert "KRmseSafetyStock" in output and "SqrtHorizonSafetyStock" in output, output
        assert "+100.00" in output, output  # 200.0 - 100.0 total_cost delta


def test_diff_reports_missing_strategy():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "results.jsonl"
        _write_two_strategy_log(log_path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            diff_strategies(log_path, "KRmseSafetyStock", "NoSuchStrategy")
        assert "no logged runs for strategy 'nosuchstrategy'" in buf.getvalue().lower()


if __name__ == "__main__":
    test_missing_log_prints_friendly_message()
    test_filters_by_strategy()
    test_diff_shows_both_strategies_and_correct_delta()
    test_diff_reports_missing_strategy()
    print("ok")
