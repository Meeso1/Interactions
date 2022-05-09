from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import RectBivariateSpline   # type: ignore
from typing import Any, Tuple


class TimeVariantData(ABC):

    def __init__(self):
        self._dt: CompoundData | None = None

    @abstractmethod
    def val(self, time: float) -> Any:
        pass

    @property
    def current(self) -> Any:
        return self.val(self.range[1])

    @property
    def dt(self) -> CompoundData:
        if self._dt is None:
            self._dt = self._get_dt()
        return self._dt

    @property
    @abstractmethod
    def range(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def _get_dt(self) -> CompoundData:
        pass

    def __add__(self, other) -> CompoundData:
        if isinstance(other, TimeVariantData):
            return CompoundDataAdd(self, other)
        else:
            return CompoundDataAddConst(self, other)

    def __radd__(self, other) -> CompoundData:
        return self + other

    def __neg__(self) -> CompoundData:
        return CompoundDataNeg(self)

    def __sub__(self, other) -> CompoundData:
        return self + (-other)

    def __rsub__(self, other) -> CompoundData:
        return (-self) + other

    def __mul__(self, other) -> CompoundData:
        if isinstance(other, TimeVariantData):
            return CompoundDataMul(self, other)
        else:
            return CompoundDataMulConst(self, other)

    def __rmul__(self, other) -> CompoundData:
        return self * other

    def __truediv__(self, other) -> CompoundData:
        if isinstance(other, TimeVariantData):
            return CompoundDataDiv(self, other)
        else:
            return CompoundDataMulConst(self, 1/other)

    def __pow__(self, power: int, modulo=None) -> CompoundData:
        if modulo is not None:
            raise ValueError("Idk what to do with modulo arg")
        return CompoundDataPowConst(self, power)


# Data that represents a composition (eg. sum) of other data
class CompoundData(TimeVariantData, ABC):
    pass


class CompoundDataAdd(CompoundData):

    def __init__(self, a: TimeVariantData, b: TimeVariantData):
        super().__init__()
        self.a: TimeVariantData = a
        self.b: TimeVariantData = b

    def val(self, time: float) -> Any:
        return self.a.val(time) + self.b.val(time)

    def _get_dt(self) -> CompoundData:
        return self.a.dt + self.b.dt

    @property
    def range(self) -> Tuple[float, float]:
        min_t = max(self.a.range[0], self.b.range[0])
        max_t = min(self.a.range[1], self.b.range[1])
        return min_t, max_t


class CompoundDataNeg(CompoundData):

    def __init__(self, a: TimeVariantData):
        super().__init__()
        self.a: TimeVariantData = a

    def val(self, time: float) -> Any:
        return -self.a.val(time)

    def _get_dt(self) -> CompoundData:
        return -self.a.dt

    @property
    def range(self) -> Tuple[float, float]:
        return self.a.range


class CompoundDataMul(CompoundData):

    def __init__(self, a: TimeVariantData, b: TimeVariantData):
        super().__init__()
        self.a: TimeVariantData = a
        self.b: TimeVariantData = b

    def val(self, time: float) -> Any:
        return self.a.val(time) * self.b.val(time)

    def _get_dt(self) -> CompoundData:
        return self.a.dt * self.b + self.a * self.b.dt

    @property
    def range(self) -> Tuple[float, float]:
        min_t = max(self.a.range[0], self.b.range[0])
        max_t = min(self.a.range[1], self.b.range[1])
        return min_t, max_t


class CompoundDataDiv(CompoundData):

    def __init__(self, a: TimeVariantData, b: TimeVariantData):
        super().__init__()
        self.a: TimeVariantData = a
        self.b: TimeVariantData = b

    def val(self, time: float) -> Any:
        return self.a.val(time) / self.b.val(time)

    def _get_dt(self) -> CompoundData:
        return (self.a.dt * self.b - self.a * self.b.dt) / (self.b * self.b)

    @property
    def range(self) -> Tuple[float, float]:
        min_t = max(self.a.range[0], self.b.range[0])
        max_t = min(self.a.range[1], self.b.range[1])
        return min_t, max_t


class CompoundDataAddConst(CompoundData):

    def __init__(self, a: TimeVariantData, c: Any):
        super().__init__()
        self.a: TimeVariantData = a
        self.c: Any = c

    def val(self, time: float) -> Any:
        return self.a.val(time) + self.c

    def _get_dt(self) -> CompoundData:
        return self.a.dt

    @property
    def range(self) -> Tuple[float, float]:
        return self.a.range


class CompoundDataMulConst(CompoundData):

    def __init__(self, a: TimeVariantData, c: Any):
        super().__init__()
        self.a: TimeVariantData = a
        self.c: Any = c

    def val(self, time: float) -> Any:
        return self.a.val(time) * self.c

    def _get_dt(self) -> CompoundData:
        return self.a.dt * self.c

    @property
    def range(self) -> Tuple[float, float]:
        return self.a.range


class CompoundDataPowConst(CompoundData):

    def __init__(self, a: TimeVariantData, c: int):
        super().__init__()
        self.a: TimeVariantData = a
        self.c: int = c

    def val(self, time: float) -> Any:
        v = self.a.val(time)
        res = 1
        if self.c >= 0:
            for i in range(self.c):
                res *= v
        else:
            for i in range(-self.c):
                res /= v
        return res

    def _get_dt(self) -> CompoundData:
        return (self.a ** (self.c - 1)) * self.a.dt * self.c

    @property
    def range(self) -> Tuple[float, float]:
        return self.a.range


# Default derivative for arithmetic data (additive and multiplicative with floats)
class CompoundDataDerivative(CompoundData):
    h: float = np.cbrt(np.finfo(float).eps)

    def __init__(self, a):
        super().__init__()
        self.a: TimeVariantData = a

    def val(self, time: float) -> Any:
        if self.range[0] < time or self.range[1] < time:
            raise ValueError(f"time argument not in range: {time} (range: [{self.range[0]}, {self.range[1]}])")
        if self.range[0] >= self.range[1]:
            raise ValueError("More than one correct time argument needed to compute derivative")

        if time + self.h <= self.range[1] and time - self.h >= self.range[0]:
            return (self.a.val(time + self.h) - self.a.val(time - self.h))/(2 * self.h)
        elif time + self.h <= self.range[1]:
            return (self.a.val(time + self.h) - self.a.val(time)) / self.h
        elif time - self.h >= self.range[0]:
            return (self.a.val(time) - self.a.val(time - self.h)) / self.h
        else:
            h = min(self.range[1] - time, time - self.range[0])
            if h > 0:
                return (self.a.val(time + h) - self.a.val(time - h))/(2 * h)
            else:
                return (self.a.val(self.range[1]) - self.a.val(self.range[1]))/(self.range[1] - self.range[0])

    def _get_dt(self) -> CompoundData:
        return CompoundDataDerivative(self)

    @property
    def range(self) -> Tuple[float, float]:
        return self.a.range
