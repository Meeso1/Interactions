from type_declarations import *
from typing import Any, List
from DeserializationError import DeserializationError
from math_primitives.Vector import Vector


# If schema validation passed, json_object is a list[int, int]
def deserialize(json_object: Any) -> Vector:
    return Vector(json_object)
