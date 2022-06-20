from type_declarations import *
from ValueType import ValueType
from math_primitives.Vector import Vector


class Vector2Type(ValueType):

    def __init__(self):
        super(Vector2Type, self).__init__(
            name="Vector2",
            val_type=Vector,
            zero=lambda: Vector(2)
        )
