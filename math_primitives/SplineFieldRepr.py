from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline   # type: ignore
from typing import Union, Type, final, Final, Callable, Any, Tuple, TypeAlias, Optional
from .FieldValueRepr import FieldValueRepr, CordVal, ValueFunc, Values
from .Derivatives import derivative_grid

ValueFuncSpline: TypeAlias = Callable[[CordVal, CordVal, Optional[int], Optional[int]], Values]


@final
class SplineFieldRepr(FieldValueRepr):

    default_res: Final[Tuple[int, int]] = (20, 20)

    def __init__(self, size: Tuple[float, float], res: Tuple[int, int] = default_res):
        self.data: NDArray[np.float64] = np.zeros(res)

        self.x_cords: NDArray[np.float64] = np.linspace(-size[0]/2, size[0]/2, res[0])
        self.y_cords: NDArray[np.float64] = np.linspace(-size[1]/2, size[1]/2, res[1])

        self.f: ValueFuncSpline = self._make_func()

    @classmethod
    def from_values(cls, size: Tuple[float, float], data: NDArray[np.float64]) -> SplineFieldRepr:
        if len(data.shape) != 2:
            raise ValueError("data must be a 2d array")

        new = cls(size, (data.shape[0], data.shape[1]))
        new.data = data
        new.f = new._make_func()
        return new

    def get_value(self, x: CordVal, y: CordVal) -> Values:
        return self.f(x, y, 0, 0)

    def derivative(self, x: CordVal, y: CordVal, dx: int = 0, dy: int = 0) -> Values:
        if 0 <= dx <= 2 and 0 <= dy <= 2:
            return self.f(x, y, dx, dy)
        else:
            return derivative_grid(self.f, x, y, dx, dy)

    def add(self, f: ValueFunc) -> None:
        self.data += f(self.x_cords, self.y_cords)
        self.f = self._make_func()

    def get_data(self) -> NDArray[np.float64] | float:
        return self(self.x_cords, self.y_cords)

    def _make_func(self) -> ValueFuncSpline:
        return RectBivariateSpline(self.x_cords, self.y_cords, self.data)