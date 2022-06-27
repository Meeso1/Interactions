from type_declarations import *
from typing import Protocol, Any, TypeVar

T = TypeVar('T')


class Deserializable(Protocol[T]):

    @staticmethod
    def deserialize(json_object: Any) -> T:
        ...
