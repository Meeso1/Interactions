from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Union, final, Callable, Any, TypeAlias, Tuple
from .Vector import Vector2

CordVal: TypeAlias = Union[float, NDArray[np.float64]]
Values: TypeAlias = Union[float, NDArray[np.float64]]
ValueFunc: TypeAlias = Callable[[CordVal, CordVal], Values]


class FieldValueRepr(ABC):

    @abstractmethod
    def get_size(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_value(self, x: CordVal, y: CordVal) -> Values:
        pass

    @abstractmethod
    def add(self, f: ValueFunc) -> None:
        pass

    @abstractmethod
    def get_data(self) -> Any:
        pass

    @abstractmethod
    def derivative(self, x: CordVal, y: CordVal, dx: int = 0, dy: int = 0) -> Values:
        pass

    @final
    def __call__(self, x: CordVal, y: CordVal) -> Values:
        return self.get_value(x, y)

    @final
    def __iadd__(self, f: ValueFunc) -> FieldValueRepr:
        self.add(f)
        return self

    def dx(self, x: CordVal, y: CordVal) -> Values:
        return self.derivative(x, y, dx=1, dy=0)

    def dy(self, x: CordVal, y: CordVal) -> Values:
        return self.derivative(x, y, dx=0, dy=1)

    def dx2(self, x: CordVal, y: CordVal) -> Values:
        return self.derivative(x, y, dx=2, dy=0)

    def dy2(self, x: CordVal, y: CordVal) -> Values:
        return self.derivative(x, y, dx=0, dy=2)

    def dxy(self, x: CordVal, y: CordVal) -> Values:
        return self.derivative(x, y, dx=1, dy=1)

    def gradient(self, x: float, y: float) -> Vector2:
        return Vector2([self.dx(x, y), self.dy(x, y)])
