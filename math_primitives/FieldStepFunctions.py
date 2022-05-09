from type_declarations.Types import *
from math_primitives.FieldValueRepr import ValueFunc


# Required steps: 2
def euler_field(derivatives: List[TimedVal[DerivativeFunc]], time: float | None = None) -> ValueFunc:

    if time is None:
        time = derivatives[-1].time

    curr = derivatives[1]
    prev = derivatives[0]
    for i in range(len(derivatives) - 1):
        if derivatives[-i-2].time < time <= derivatives[-i-1].time:
            curr = derivatives[-i-1]
            prev = derivatives[-i-2]
            break

    return lambda x, y: curr.val(x, y) * (curr.time - prev.time)
