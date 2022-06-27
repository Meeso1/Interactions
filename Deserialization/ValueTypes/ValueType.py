from __future__ import annotations

from Deserialization.Deserializable import Deserializable
from type_declarations import *
from typing import TypeVar, Generic, Callable, Type, Any, Dict, Iterable

T = TypeVar("T", bound=Deserializable)


class ValueType(Generic[T]):
    types: Dict[str, ValueType] = {}

    def __init__(self, name: str, val_type: Type[T], checker: Callable[[Any], bool], zero: Callable[[], T]):
        self.name: str = name
        self.type: Type[T] = val_type
        self.checker: Callable[[Any], bool] = checker
        self.zero: Callable[[], T] = zero

    @staticmethod
    def register(new_type: ValueType):
        ValueType.types[new_type.name] = new_type
