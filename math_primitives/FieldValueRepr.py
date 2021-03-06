from __future__ import annotations
from abc import ABC, abstractmethod
from typing import final, Any, Tuple
from type_declarations.Arithmetic import Arithmetic
from .Vector import Vector2
from type_declarations.Types import *


class FieldValueRepr(ABC):

    @property
    @abstractmethod
    def const(self) -> bool:
        pass

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
    def get_data(self) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def copy(self) -> FieldValueRepr:
        pass

    @abstractmethod
    def derivative(self, x: CordVal, y: CordVal, dx: int = 0, dy: int = 0) -> Values:
        pass

    @classmethod
    @final
    def from_values(cls, size: Tuple[float, float], data: NDArray[np.float64], const: bool = False) -> FieldValueRepr:
        return cls._from_values(size, data, const)

    @staticmethod
    @abstractmethod
    def _from_values(size: Tuple[float, float], data: NDArray[np.float64], const: bool) -> FieldValueRepr:
        pass

    @final
    def __call__(self, x: CordVal, y: CordVal) -> Values:
        return self.get_value(x, y)

    # Arithmetic operators:

    @abstractmethod
    def __add__(self, other: FieldValueRepr) -> FieldValueRepr:
        pass

    @abstractmethod
    def __neg__(self) -> FieldValueRepr:
        pass

    @abstractmethod
    def __mul__(self, other: float) -> FieldValueRepr:
        pass

    @final
    def __radd__(self, other: FieldValueRepr) -> FieldValueRepr:
        return self + other

    @final
    def __sub__(self, other: FieldValueRepr) -> FieldValueRepr:
        return self + (-other)

    @final
    def __rsub__(self, other: FieldValueRepr) -> FieldValueRepr:
        return self - other

    @final
    def __rmul__(self, other: float) -> FieldValueRepr:
        return self * other

    # Analytical properties:

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
