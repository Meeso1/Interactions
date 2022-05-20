import numpy as np
from typing import TypeVar, TypeAlias, Callable, List, Optional
from type_declarations.Arithmetic import Arithmetic
from numpy.typing import NDArray
from math_primitives.Vector import Vector
from math_primitives.TimedVal import TimedVal

N = TypeVar("N", bound=Arithmetic)

CordVal: TypeAlias = float | NDArray[np.float64]
Values: TypeAlias = float | NDArray[np.float64]
ValueFunc: TypeAlias = Callable[[CordVal, CordVal], Values]
DerivativeFunc: TypeAlias = Callable[[CordVal, CordVal], Values]
# Type of field derivative function.
# For arguments x = [x1, x2, x3, ...] and y = [y1, y2, y3, ...]
# returns an array of derivatives (d/dt) at points x*y (vector product)

NumericStepFunc: TypeAlias = Callable[[List[TimedVal[N]], Optional[float]], N]
# eg.: def euler(derivatives: List[TimedVal], time: float | None = None) -> N:
#           return df(time ?? derivatives[-1].time) * dt <- inferred from derivatives list

FieldStepFunc: TypeAlias = Callable[[List[TimedVal[DerivativeFunc]], Optional[float]], ValueFunc]
# eg.: def euler_field(derivatives: List[TimedVal], time: float | None = None) -> FieldValueRepr:
#           return df(time ?? derivatives[-1].time) * dt <- inferred from derivatives list
