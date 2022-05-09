import numpy as np
from typing import TypeVar, TypeAlias, Callable, List, Optional
from numpy.typing import NDArray
from math_primitives.Vector import Vector
from math_primitives.TimedVal import TimedVal

N = TypeVar("N", float, Vector, NDArray[np.float64])

CordVal: TypeAlias = float | NDArray[np.float64]
Values: TypeAlias = float | NDArray[np.float64]
ValueFunc: TypeAlias = Callable[[CordVal, CordVal], Values]
DerivativeFunc: TypeAlias = Callable[[CordVal, CordVal], Values]

NumericStepFunc: TypeAlias = Callable[[List[TimedVal[N]], Optional[float]], N]
# eg.: def euler(derivatives: List[TimedVal], time: float | None = None) -> N:
#           return df(time ?? derivatives[-1].time) * dt <- inferred from derivatives list

FieldStepFunc: TypeAlias = Callable[[List[TimedVal[DerivativeFunc]], Optional[float]], ValueFunc]
# eg.: def euler_field(derivatives: List[TimedVal], time: float | None = None) -> FieldValueRepr:
#           return df(time ?? derivatives[-1].time) * dt <- inferred from derivatives list
