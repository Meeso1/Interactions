from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline   # type: ignore
from typing import final, Final, Callable, Tuple, TypeAlias, Any
from .FieldValueRepr import FieldValueRepr, CordVal, ValueFunc, Values
from .Derivatives import derivative_grid
from .SplineFieldRepr import ValueFuncSpline
from .Vector import is_array

default_res: Tuple[int, int] = (20, 20)


@final
class ConstantFieldRepr(FieldValueRepr):

    def __init__(self, size: Tuple[float, float], res: Tuple[int, int] = default_res):
        self.data: NDArray[np.float64] = np.zeros(res)
        self.size: Tuple[float, float] = size

        self.x_cords: NDArray[np.float64] = np.linspace(-size[0]/2, size[0]/2, res[0])
        self.y_cords: NDArray[np.float64] = np.linspace(-size[1]/2, size[1]/2, res[1])

        self.f: ValueFuncSpline = self._make_func()

    def _make_func(self) -> ValueFuncSpline:
        return RectBivariateSpline(self.x_cords, self.y_cords, self.data)

    def get_size(self) -> Tuple[float, float]:
        return self.size

    def get_value(self, x: CordVal, y: CordVal) -> Values:
        v = self.f(x, y)
        return v[0, 0] if is_array(v) and v.size == 1 else v

    def add(self, f: ValueFunc) -> None:
        vals = f(self.x_cords, self.y_cords)
        if isinstance(vals, float):
            if vals != 0:
                raise RuntimeError("Cannot add() a non-zero value to ConstantFieldRepr")
        elif not (vals == 0).all():
            raise RuntimeError("Cannot add() a non-zero value to ConstantFieldRepr")

    def get_data(self) -> Any:
        return self.data.copy()

    def copy(self) -> FieldValueRepr:
        return ConstantFieldRepr.from_values(self.get_size(), self.data.copy())

    def derivative(self, x: CordVal, y: CordVal, dx: int = 0, dy: int = 0) -> Values:
        return derivative_grid(self.f, x, y, dx, dy)

    @staticmethod
    def _from_values(size: Tuple[float, float], data: NDArray[np.float64]) -> FieldValueRepr:
        if len(data.shape) != 2:
            raise ValueError("data must be a 2d array")

        new = ConstantFieldRepr(size, (data.shape[0], data.shape[1]))
        new.data = data
        new.f = new._make_func()
        return new
