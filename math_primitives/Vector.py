from __future__ import annotations
import numpy as np
from abc import ABC
from typing import Type, Any, Dict, TypeGuard, Final, TypeAlias
from numpy.typing import NDArray


def is_num(a: Any) -> TypeGuard[int | float]:
    return isinstance(a, (int, float))


def is_array(a: Any) -> TypeGuard[NDArray[np.float64]]:
    return isinstance(a, np.ndarray) and a.dtype in (float, np.float64, int)


def is_vector(v: Any) -> TypeGuard[Vector]:
    return isinstance(v, Vector)


class Vector:

    def __init__(self, data: Any = 2) -> None:
        self.data: NDArray[np.float64]

        # Make zero Vector with given size
        if isinstance(data, int):
            if data > 0:
                _data = np.zeros((data,), dtype=np.float64)
            else:
                raise ValueError("Size of the Vector must be a positive integer")
        # Copy from Vector
        elif is_vector(data):
            _data = data.data
        # Vector with cords in numpy array
        elif is_array(data):
            _data = data
        # Try to convert to array
        else:
            try:
                _data = np.array(data, dtype=np.float64)
            except BaseException:
                raise Exception(f"Cannot create a Vector from {type(data)}")

        self.data = _data

        if self.data.size == 0:
            raise ValueError(f"""Cannot create a Vector from a 0-sized numpy array: {str(data)}""")

        self.n: Final[int] = len(self.data)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Vector{self.n}({str(self.data)})"

    def __add__(self, other: Any) -> Vector:
        if is_vector(other) and len(self) == len(other):
            return Vector(self.data + other.data)
        return Vector(self.data + other)

    def __radd__(self, other: Any) -> Vector:
        return self.__add__(other)

    def __neg__(self) -> Vector:
        return Vector(-self.data)

    def __sub__(self, other: Any) -> Vector:
        if is_vector(other):
            return self + (-other)
        return self + (-np.array(other, dtype=np.float64))

    def __rsub__(self, other: Any) -> Vector:
        return self.__sub__(other)

    def __mul__(self, other: Any) -> Vector | float:
        if is_num(other):
            return Vector(self.data * other)
        return self.dot(other)

    def __rmul__(self, other: Any) -> Vector | float:
        return self.__mul__(other)

    def __truediv__(self, other: int | float) -> Vector:
        if is_num(other):
            if other == 0:
                raise ArithmeticError()
            r = self * (1 / other)
            if is_vector(r):  # Always true, added to pass type check
                return r
        raise NotImplementedError()

    def __abs__(self) -> float:
        return self.length()

    def __eq__(self, other: object) -> bool:
        try:
            vec = Vector(other)
            return np.array_equal(self.data, vec.data)
        except BaseException:
            return False

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> float:
        if not isinstance(i, int) or not (0 <= i < self.n):
            raise IndexError(f"Incorrect index for Vector{self.n}: {i}")
        return self.data[i]

    def __setitem__(self, i: int, val: int | float) -> None:
        if not isinstance(i, int) or not (0 <= i < self.n):
            raise IndexError(f"Incorrect index for Vector{self.n}: {i}")
        if not is_num(val):
            raise TypeError(f"Cannot assign {type(val)} to Vector coordinate")
        self.data[i] = val

    def length(self) -> float:
        return np.sqrt(self.dot(self))

    def dot(self, other: Any) -> float:
        if is_vector(other) and len(self) == len(other):
            return sum(other.data * self.data)
        v = Vector(other)
        if len(v) != len(self):
            raise ValueError(f"Vector lengths don't match: {len(v)} != {len(self)}")
        return sum(v.data * self.data)

    def normalize(self) -> Vector:
        return self / self.length() if not self.length() == 0 else Vector(len(self))

    @classmethod
    def distance(cls, a: Any, b: Any) -> float:
        try:
            _a = Vector(a)
            _b = Vector(b)
        except BaseException:
            raise Exception(f"Cannot convert types {type(a)}, {type(b)} to Vector")
        return (_a - _b).length()

    @property
    def x(self):
        return self[0]

    @x.getter
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    # y property
    @property
    def y(self):
        return self[1] if self.n > 1 else None

    @y.getter
    def y(self):
        return self[1] if self.n > 1 else None

    @y.setter
    def y(self, value):
        if len(self) < 2:
            raise NotImplementedError()
        self[1] = value

    # z property
    @property
    def z(self):
        return self[2] if self.n > 2 else None

    @z.getter
    def z(self):
        return self[2] if self.n > 2 else None

    @z.setter
    def z(self, value):
        if len(self) < 3:
            raise NotImplementedError()
        self[2] = value


# Commonly used vectors
Vector2: TypeAlias = Vector
Vector3: TypeAlias = Vector
