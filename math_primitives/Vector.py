from __future__ import annotations
import numpy as np
from typing import Union, Type


class VectorBase:

    vector_classes = {}


def Vector(n: int) -> Type:
    if not isinstance(n, int) or n <= 0:
        raise Exception(f"Argument must be a positive integer")

    if n in VectorBase.vector_classes:
        return VectorBase.vector_classes[n]

    class VectorN(VectorBase):
        def __init__(self, data=np.zeros((n,), dtype=float)) -> None:
            _data = data
            if isinstance(data, VectorN):
                _data = data.data
            # Try to convert to array
            if not isinstance(data, np.ndarray):
                try:
                    _data = np.array(data)
                except BaseException:
                    raise Exception(f"Cannot create a Vector{n} from {type(data)}")
            # Check shape
            if not _data.shape == (n,):
                raise Exception(f"Cannot create a Vector{n} from an array of shape {_data.shape}")
            self.data = _data

        def __repr__(self) -> str:
            return self.__str__()

        def __str__(self) -> str:
            return f"Vector2({str(self.data)})"

        def __add__(self, other) -> VectorN:
            if isinstance(other, VectorN):
                return VectorN(self.data + other.data)
            return VectorN(self.data + other)

        def __radd__(self, other) -> VectorN:
            return self.__add__(other)

        def __neg__(self) -> VectorN:
            return VectorN(-self.data)

        def __sub__(self, other) -> VectorN:
            return self.__add__(-other)

        def __rsub__(self, other) -> VectorN:
            return self.__sub__(other)

        def __mul__(self, other) -> Union[VectorN, float]:
            if isinstance(other, (int, float)):
                return VectorN(self.data * other)
            return self.dot(other)

        def __rmul__(self, other) -> Union[VectorN, float]:
            return self.__mul__(other)

        def __truediv__(self, other) -> VectorN:
            if not isinstance(other, (int, float)):
                raise NotImplementedError()
            if other == 0:
                raise ArithmeticError()
            return self * (1 / other)

        def __abs__(self) -> float:
            return self.length()

        def __eq__(self, other):
            _other = other
            if not isinstance(other, VectorN):
                _other = VectorN(other)
            return np.array_equal(self.data, _other.data)

        def __len__(self) -> int:
            return n

        def __getitem__(self, i) -> float:
            if not isinstance(i, int) or not (0 <= i < n):
                raise IndexError(f"Incorrect index for Vector{n}: {i}")
            return self.data[i]

        def __setitem__(self, i, val) -> None:
            if not isinstance(i, int) or not (0 <= i < n):
                raise IndexError(f"Incorrect index for Vector{n}: {i}")
            if not isinstance(val, (int, float)):
                raise TypeError(f"Cannot assign {type(val)} to VectorN coordinate")
            self.data[i] = val

        def length(self) -> float:
            return np.sqrt(self.dot(self))

        def dot(self, other) -> float:
            _other = other
            if not isinstance(other, VectorN):
                _other = VectorN(other)
            return sum(_other.data * self.data)

        def normalize(self) -> VectorN:
            return self / self.length() if not self.length() == 0 else VectorN()

        @classmethod
        def distance(cls, a, b) -> float:
            try:
                _a = VectorN(a)
                _b = VectorN(b)
            except BaseException:
                raise Exception(f"Cannot convert types {type(a)}, {type(b)} to Vector{n}")
            return (_a-_b).length()

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
        if n >= 2:
            @property
            def y(self):
                return self[1]

            @y.getter
            def y(self):
                return self[1]

            @y.setter
            def y(self, value):
                self[1] = value

        # z property
        if n >= 3:
            @property
            def z(self):
                return self[2]

            @z.getter
            def z(self):
                return self[2]

            @z.setter
            def z(self, value):
                self[2] = value

    VectorBase.vector_classes[n] = VectorN
    return VectorN


# Commonly used vectors
Vector2: Type = Vector(2)
Vector3: Type = Vector(3)
