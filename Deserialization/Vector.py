from type_declarations import *
from typing import Any, List
from DeserializationError import DeserializationError
from math_primitives.Vector import Vector


def deserialize(json_object: Any, validated: bool = False) -> Vector:
    if not validated:
        if not valid(json_object):
            raise DeserializationError(Vector, json_object)
    return Vector(json_object)


def valid(json_object: Any) -> bool:
    # [number, number] => True
    return isinstance(json_object, List) \
           and len(json_object) == 2 \
           and isinstance(json_object[0], int | float) \
           and isinstance(json_object[1], int | float)
