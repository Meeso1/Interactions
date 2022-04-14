from __future__ import annotations
import numpy as np
from typing import TypeAlias, Callable, Optional, Any, cast
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]
CordVal: TypeAlias = float | NDArray[np.float64]
Values: TypeAlias = float | NDArray[np.float64]
ValueFunc: TypeAlias = Callable[[CordVal, CordVal], Values]

DerivativeInternalFunc: TypeAlias = \
    Callable[[CordVal, CordVal, Callable[..., Values]], Values]


def assert_type(f: Callable[..., Values]) -> DerivativeInternalFunc:
    return cast(DerivativeInternalFunc, f)


def derivative_func(func: ValueFunc, dx: int = 0, dy: int = 0) -> ValueFunc:
    h = np.cbrt(np.finfo(float).eps)

    f1: DerivativeInternalFunc = assert_type(lambda x, y, f=func: func(x, y))
    for i in range(dx):
        f1 = assert_type(lambda x, y, f=f1: (f(x + h, y) - f(x - h, y)) / (2 * h))
    for i in range(dy):
        f1 = assert_type(lambda x, y, f=f1: (f(x, y + h) - f(x, y - h)) / (2 * h))

    f2: DerivativeInternalFunc = assert_type(lambda x, y, f=None: func(x, y))
    for i in range(dy):
        f2 = assert_type(lambda x, y, f=f2: (f(x + h, y) - f(x - h, y)) / (2 * h))
    for i in range(dx):
        f2 = assert_type(lambda x, y, f=f2: (f(x, y + h) - f(x, y - h)) / (2 * h))

    res: ValueFunc = lambda x, y: (f1(x, y) + f2(x, y))/2   # type: ignore
    return res


def derivative_grid(func: ValueFunc, x: CordVal, y: CordVal, dx: int = 0, dy: int = 0) -> Values:
    return derivative_func(func, dx, dy)(x, y)
