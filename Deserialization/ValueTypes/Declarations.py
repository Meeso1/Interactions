from type_declarations import *
from ValueType import ValueType
from math_primitives.Vector import Vector


def declare():
    ValueType.register(ValueType(
        name="Vector2",
        val_type=Vector,
        checker=lambda val: isinstance(val, Vector) and len(val) == 2,
        zero=lambda: Vector(2))
    )
    ValueType.register(ValueType(
        name="float",
        val_type=float,
        checker=lambda val: isinstance(val, float),
        zero=lambda: 0)
    )
