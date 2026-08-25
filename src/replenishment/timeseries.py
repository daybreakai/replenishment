"""Shared forecast/actuals abstraction: a list or callable, with
extend-last-value lookup past the end of a finite series."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeSeries:
    _values: list[float] | None = field(default=None, repr=False)
    _model: Callable[[int], float] | None = field(default=None, repr=False)

    def __init__(self, _values=None, _model=None, *, values=None, model=None) -> None:
        # values=/model= are the public spelling; TimeSeries.from_values/from_callable
        # remain the recommended constructors, this just makes TimeSeries(values=[...])
        # work instead of raising "unexpected keyword argument 'values'".
        object.__setattr__(self, "_values", _values if _values is not None else values)
        object.__setattr__(self, "_model", _model if _model is not None else model)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (self._values is None) == (self._model is None):
            raise ValueError(
                "TimeSeries requires exactly one of values or model "
                "(use TimeSeries.from_values([...]) or TimeSeries.from_callable(fn), "
                "or TimeSeries(values=[...]) / TimeSeries(model=fn))."
            )

    @classmethod
    def from_values(cls, values: list[float]) -> "TimeSeries":
        return cls(_values=list(values))

    @classmethod
    def from_callable(cls, model: Callable[[int], float]) -> "TimeSeries":
        return cls(_model=model)

    def value_at(self, period: int) -> float:
        if period < 0:
            raise IndexError("TimeSeries period out of range.")
        if self._model is not None:
            return self._model(period)
        if not self._values:
            raise IndexError("TimeSeries period out of range.")
        if period >= len(self._values):
            return self._values[-1]
        return self._values[period]

    def sum_over(self, start: int, horizon: int) -> float:
        if horizon <= 0:
            return 0.0
        return sum(self.value_at(start + offset) for offset in range(horizon))
