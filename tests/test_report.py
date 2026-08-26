import math

from replenishment.report import PolicyRun, build_report
from replenishment.strategies.resolver import resolve_safety_stock_strategy
from replenishment.simulation import SimulationResult, SimulationSummary


def _result(fill_rate: float, total_demand: int = 100) -> SimulationResult:
    fulfilled = 0 if math.isnan(fill_rate) else int(total_demand * fill_rate)
    summary = SimulationSummary(
        total_demand=total_demand, total_fulfilled=fulfilled,
        total_backorders=total_demand - fulfilled, fill_rate=fill_rate,
        average_on_hand=50.0, holding_cost=50.0, stockout_cost=0.0,
        ordering_cost=10.0, total_cost=60.0,
    )
    return SimulationResult(snapshots=[], summary=summary, ending_pipeline=[])


def test_build_report_marks_healthy_policy():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku-1", result=_result(fill_rate=0.98), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert len(report.records) == 1
    assert report.records[0].health_status == "Healthy"
    assert report.records[0].safety_stock_method == "SqrtHorizonSafetyStock"
    assert report.records[0].degraded is False


def test_build_report_flags_understock_risk_below_90pct_fill_rate():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    entry = PolicyRun(label="sku-2", result=_result(fill_rate=0.80), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert report.records[0].health_status == "Understock Risk"
    assert report.records[0].degraded is True


def test_build_report_no_data_status_for_zero_demand():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    entry = PolicyRun(label="sku-3", result=_result(fill_rate=1.0, total_demand=0), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert report.records[0].health_status == "No Data"


def test_build_report_summary_counts_and_skipped_stages():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entries = [
        PolicyRun(label="sku-1", result=_result(fill_rate=0.98), resolved_safety_stock=resolved),
        PolicyRun(label="sku-2", result=_result(fill_rate=0.98), resolved_safety_stock=resolved),
    ]
    report = build_report(entries)
    assert report.summary.total_policies == 2
    assert report.summary.status_counts["Healthy"] == 2
    assert "overstock" in " ".join(report.summary.stages_skipped).lower()


def test_to_markdown_renders_a_table_with_all_labels():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku-1", result=_result(fill_rate=0.98), resolved_safety_stock=resolved)
    report = build_report([entry])
    markdown = report.to_markdown()
    assert "sku-1" in markdown
    assert "Healthy" in markdown


def test_policy_runs_from_portfolio_zips_results_with_resolved_strategies():
    from replenishment.portfolio import PortfolioResult
    from replenishment.report import policy_runs_from_portfolio

    resolved_a = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    resolved_b = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    portfolio_result = PortfolioResult(results={
        "sku-a": _result(fill_rate=0.98),
        "sku-b": _result(fill_rate=0.80),
    })
    runs = policy_runs_from_portfolio(
        portfolio_result, {"sku-a": resolved_a, "sku-b": resolved_b},
    )
    by_label = {r.label: r for r in runs}
    assert by_label["sku-a"].resolved_safety_stock is resolved_a
    assert by_label["sku-b"].resolved_safety_stock is resolved_b
    report = build_report(runs)
    assert {r.label for r in report.records} == {"sku-a", "sku-b"}


def test_build_report_boundary_fill_rate_90_percent_is_healthy():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku-boundary", result=_result(fill_rate=0.90), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert report.records[0].health_status == "Healthy"


def test_build_report_understock_threshold_is_overridable():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku-override", result=_result(fill_rate=0.95), resolved_safety_stock=resolved)
    report = build_report([entry], understock_fill_rate_threshold=0.99)
    assert report.records[0].health_status == "Understock Risk"


def test_policy_runs_from_portfolio_missing_key_degrades_instead_of_raising():
    from replenishment.portfolio import PortfolioResult
    from replenishment.report import policy_runs_from_portfolio

    resolved_a = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    portfolio_result = PortfolioResult(results={
        "sku-a": _result(fill_rate=0.98),
        "sku-b": _result(fill_rate=0.80),
    })
    runs = policy_runs_from_portfolio(portfolio_result, {"sku-a": resolved_a})  # sku-b missing on purpose
    by_label = {r.label: r for r in runs}
    assert by_label["sku-a"].resolved_safety_stock is resolved_a
    assert by_label["sku-b"].resolved_safety_stock.degraded is True
    assert "sku-b" in by_label["sku-b"].resolved_safety_stock.reason


def test_to_markdown_includes_stages_skipped_and_degradation_reason():
    resolved = resolve_safety_stock_strategy(has_actuals=False, periods_observed=0)
    entry = PolicyRun(label="sku-1", result=_result(fill_rate=0.80), resolved_safety_stock=resolved)
    report = build_report([entry])
    markdown = report.to_markdown()
    assert "overstock" in markdown.lower()
    assert "Stages skipped" in markdown


def test_build_report_empty_list():
    report = build_report([])
    assert report.records == []
    assert report.summary.total_policies == 0
    assert report.summary.status_counts == {}


def test_to_markdown_escapes_pipe_in_label():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku|weird", result=_result(fill_rate=0.98), resolved_safety_stock=resolved)
    report = build_report([entry])
    markdown = report.to_markdown()
    assert "sku\\|weird" in markdown


def test_nan_fill_rate_classifies_as_no_data_not_healthy():
    resolved = resolve_safety_stock_strategy(has_actuals=True, periods_observed=10, factor=1.65)
    entry = PolicyRun(label="sku-nan", result=_result(fill_rate=float("nan")), resolved_safety_stock=resolved)
    report = build_report([entry])
    assert report.records[0].health_status == "No Data"
