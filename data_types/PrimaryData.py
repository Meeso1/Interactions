from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Generic, Any, Literal, Type

from math_primitives.QuarticSplineGenerator import QuarticSplineGenerator
from type_declarations.Types import *
from math_primitives.NumericStepFunctions import Euler
from math_primitives.TimedVal import TimedVal
from data_types.TimeVariantData import TimeVariantData, CompoundData, CompoundDataDerivative

from scipy.interpolate import RectBivariateSpline, CubicSpline   # type: ignore


class PrimaryData(TimeVariantData, ABC):

    @abstractmethod
    def add_derivative(self, p: Tuple[float, Any]):
        pass


# For numeric time-variant data, like velocity. Not for field values.
class PrimaryDataNumeric(PrimaryData, Generic[N]):

    def __init__(self,
                 val_type: Type[N],
                 step_func: NumericStepFunc = Euler(),
                 initial: N = None,             # At t = start_time
                 initial_derivative: N = None,  # At t = start_time
                 zero: Callable[[], N] | None = None,
                 start_time: float = 0,
                 update: Literal["update", "lazy", "explicit"] = "explicit",
                 create_func: Callable[[List[TimedVal[N]]], Callable[[float], N]] | None = None):
        super().__init__()

        self.start_time: float = start_time
        self.time: float = self.start_time      # Current time, based on last add_derivative() call
                                                # Updated during collapse_last_val() call
        self.zero: Callable[[], N] = zero if zero is not None else \
            (lambda: val_type() if val_type in (int, float, complex) else self._raise_no_arg_error("zero"))  # type: ignore
        self.update: Literal["update", "lazy", "explicit"] = update

        self.derivatives: List[TimedVal[N]] = [TimedVal(
            initial_derivative if initial_derivative is not None else self.zero(),
            self.time
        )]
        self.values: List[TimedVal[N]] = [TimedVal(
            initial if initial is not None else self.zero(),
            self.time
        )]

        self.step_func: NumericStepFunc = step_func
        self._default_make_func: Callable[[List[TimedVal[N]]], Callable[[float], N]] \
            = QuarticSplineGenerator(self.values, self.derivatives[0].val, self.zero())
        self.create_func: Callable[[List[TimedVal[N]]], Callable[[float], N]]
        if create_func is None:
            self.create_func = self._default_make_func
        else:
            self.create_func = create_func
        self.f: Callable[[float], N] = lambda t: \
            self.values[0].val if t == start_time else self._raise_time_arg_error(t)
        self.data_changed: bool = False     # Only used if update == "lazy"

    def _make_func(self) -> Callable[[float], N]:
        return self.create_func(self.values)

    def _raise_time_arg_error(self, time: float) -> Any:
        raise ValueError(f"time must be between start and current: {self.start_time} <= (t: {time}) <= {self.time}")

    def _raise_no_arg_error(self, arg_name: str) -> Any:
        raise ValueError(f"Following arguments are required in this context :{arg_name}")

    def val(self, time: float) -> N:
        if not self.update == "lazy" and not self.start_time <= time <= self.time:
            self._raise_time_arg_error(time)
        if self.update == "lazy":
            if self.values[-1].time <= time:
                return self.f(time)
            else:
                # this doesn't work :/
                # but I don't use it so it's fine
                i = 0
                for i in range(len(self.values)):
                    if self.values[-i-1].time >= time:
                        break
                for j in range(i, 1):
                    t = self.derivatives[-i].time
                    df = self.step_func(self.derivatives, t)
                    self.values.append(TimedVal(self.values[-1].val + df, t))
                self.time = self.derivatives[-1].time
                self.f = self._make_func()
        return self.f(time)

    def add_derivative(self, p: Tuple[float, N]) -> None:
        (time, value) = p

        if self.derivatives[-1].time == time:
            self.derivatives[-1].val += value
        else:
            self.derivatives.append(TimedVal(value, time))

        if self.update == "update":
            df = self.step_func(self.derivatives, None)
            self.time = self.derivatives[-1].time
            if self.values[-1].time == self.time:
                self.values[-1].val = self.values[-2].val + df
            else:
                self.values.append(TimedVal(self.values[-1].val + df, self.time))
            self.f = self._make_func()

    def collapse_last_val(self) -> None:
        if len(self.derivatives) < 2:
            raise AttributeError("At least 2 derivative values needed to make a step")
        self.time = self.derivatives[-1].time
        df = self.step_func(self.derivatives, None)
        self.values.append(TimedVal(self.values[-1].val + df, self.time))
        self.f = self._make_func()

    def _get_dt(self) -> CompoundData:
        return CompoundDataDerivative(self)

    @property
    def range(self) -> Tuple[float, float]:
        return self.start_time, self.time
