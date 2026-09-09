from replenishment.strategies import describe, list_order_triggers, list_safety_stock_strategies


def test_list_safety_stock_strategies_includes_known_strategy():
    assert "KRmseSafetyStock" in list_safety_stock_strategies()


def test_list_order_triggers_includes_flat_reorder_point():
    assert "FlatReorderPointTrigger" in list_order_triggers()


def test_describe_reports_factor_and_k_alias():
    params = describe("KRmseSafetyStock")["params"]
    assert "factor" in params
    assert "k" in params
