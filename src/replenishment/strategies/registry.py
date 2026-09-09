"""Read-only introspection for strategy/trigger discovery -- NOT a
construction layer. Callers still do `StrategyClass(**kwargs)` directly and
wire the result into ReplenishmentPolicy; this module only answers "what
exists, what params does it take" for an agent choosing among them.

Deliberately separate from io_.py's `_SAFETY_STOCK_STRATEGIES`/
`_FACTOR_KWARG`: that private dict only covers 6 of the strategies below
(Portfolio's narrow string-keyed sweep API, one scalar kwarg each). This
registry lists every strategy/trigger and every constructor param, for
callers going straight to the class instead of through Portfolio.
"""
from __future__ import annotations

import inspect

from replenishment.strategies.demand_buffer import DemandBufferDecorator
from replenishment.strategies.distributional_safety_stock import (
    CompoundPoissonSafetyStock,
    KingsFormulaSafetyStock,
    NegativeBinomialSafetyStock,
)
from replenishment.strategies.multiplier import MultiplierSafetyStockStrategy, NullSafetyStockStrategy
from replenishment.strategies.order_trigger import (
    FlatForecastOrderUpToTrigger,
    FlatReorderPointTrigger,
    OrderUpToTrigger,
    ReorderPointTrigger,
)
from replenishment.strategies.safety_stock import (
    FillRateSafetyStock,
    FixedErrorSafetyStock,
    KMaeSafetyStock,
    KRmseSafetyStock,
    SqrtHorizonSafetyStock,
)

SAFETY_STOCK_STRATEGIES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        SqrtHorizonSafetyStock,
        KRmseSafetyStock,
        KMaeSafetyStock,
        FixedErrorSafetyStock,
        FillRateSafetyStock,
        KingsFormulaSafetyStock,
        CompoundPoissonSafetyStock,
        NegativeBinomialSafetyStock,
        MultiplierSafetyStockStrategy,
        NullSafetyStockStrategy,
        DemandBufferDecorator,
    )
}

ORDER_TRIGGERS: dict[str, type] = {
    cls.__name__: cls
    for cls in (OrderUpToTrigger, FlatForecastOrderUpToTrigger, ReorderPointTrigger, FlatReorderPointTrigger)
}


def list_safety_stock_strategies() -> list[str]:
    return sorted(SAFETY_STOCK_STRATEGIES)


def list_order_triggers() -> list[str]:
    return sorted(ORDER_TRIGGERS)


def _resolve(name_or_class: str | type) -> type:
    if isinstance(name_or_class, type):
        return name_or_class
    for registry in (SAFETY_STOCK_STRATEGIES, ORDER_TRIGGERS):
        if name_or_class in registry:
            return registry[name_or_class]
    raise KeyError(f"Unknown strategy/trigger name: {name_or_class!r}")


def describe(name_or_class: str | type) -> dict:
    """Constructor signature for one strategy/trigger: {name, doc, params},
    params keyed by kwarg name -> {type, default, required}.

    Reads inspect.signature(cls.__init__) rather than dataclasses.fields():
    SqrtHorizonSafetyStock/KRmseSafetyStock/KMaeSafetyStock hand-write
    __init__ for factor=/k= alias resolution, so fields() would report a
    single `factor` field and silently miss the k= alias entirely.
    """
    cls = _resolve(name_or_class)
    sig = inspect.signature(cls.__init__)
    params = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        params[name] = {
            "type": str(p.annotation) if p.annotation is not inspect.Parameter.empty else None,
            "default": None if p.default is inspect.Parameter.empty else p.default,
            "required": p.default is inspect.Parameter.empty,
        }
    return {"name": cls.__name__, "doc": inspect.getdoc(cls), "params": params}
