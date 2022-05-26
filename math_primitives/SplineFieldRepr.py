from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline   # type: ignore
from typing import final, Final, Callable, Tuple, TypeAlias, cast
from .FieldValueRepr import FieldValueRepr, CordVal, ValueFunc, Values
from .Derivatives import derivative_grid
from .Vector import is_array

ValueFuncSpline: TypeAlias = Callable[..., Values]


@final
class SplineFieldRepr(FieldValueRepr):

    default_res: Final[Tuple[int, int]] = (20, 20)

    def __init__(self, size: Tuple[float, float], res: Tuple[int, int] = default_res,
                 ignore_add: bool = False):
        self.data: NDArray[np.float64] = np.zeros(res)
        self.size: Tuple[float, float] = size

        self.x_cords: NDArray[np.float64] = np.linspace(-size[0]/2, size[0]/2, res[0])
        self.y_cords: NDArray[np.float64] = np.linspace(-size[1]/2, size[1]/2, res[1])

        self.f: ValueFuncSpline = self._make_func()
        self._ignore_add: bool = ignore_add

    def __add__(self, other: FieldValueRepr) -> FieldValueRepr:
        d1 = self.get_data()
        d2 = other.get_data()
        if d1.size >= d2.size:
            return SplineFieldRepr.from_values(self.size, d1 + other(self.x_cords, self.y_cords))
        else:
            return other + self

    def __neg__(self) -> SplineFieldRepr:
        return cast(SplineFieldRepr, SplineFieldRepr.from_values(self.size, -self.get_data()))

    def __mul__(self, other: float) -> SplineFieldRepr:
        return cast(SplineFieldRepr, SplineFieldRepr.from_values(self.size, other * self.get_data()))

    @property
    def const(self) -> bool:
        return self._ignore_add

    def get_size(self) -> Tuple[float, float]:
        return self.size

    @staticmethod
    def _from_values(size: Tuple[float, float], data: NDArray[np.float64], const: bool) -> SplineFieldRepr:
        if len(data.shape) != 2:
            raise ValueError("data must be a 2d array")

        new = SplineFieldRepr(size, (data.shape[0], data.shape[1]), const)
        new.data = data
        new.f = new._make_func()
        return new

    def get_value(self, x: CordVal, y: CordVal) -> Values:
        v = self.f(x, y)
        return v[0, 0] if is_array(v) and v.size == 1 else v

    def derivative(self, x: CordVal, y: CordVal, dx: int = 0, dy: int = 0) -> Values:
        if 0 <= dx <= 2 and 0 <= dy <= 2:
            return val[0, 0] if is_array(val := self.f(x, y, dx=dx, dy=dy)) and val.size == 1 else val
        else:
            f: ValueFunc = lambda x_c, y_c: v[0, 0] if is_array(v := self.f(x_c, y_c)) and v.size == 1 else v
            return derivative_grid(f, x, y, dx, dy)

    def add(self, f: ValueFunc) -> None:
        if self._ignore_add:
            return
        self.data += f(self.x_cords, self.y_cords)
        self.f = self._make_func()

    def get_data(self) -> NDArray[np.float64]:
        return v if is_array(v := self(self.x_cords, self.y_cords)) else np.array(v)

    def copy(self) -> FieldValueRepr:
        return SplineFieldRepr.from_values(self.get_size(), self.data.copy())

    def _make_func(self) -> ValueFuncSpline:
        return RectBivariateSpline(self.x_cords, self.y_cords, self.data)
