from type_declarations import *
from typing import TypeVar, Generic, Callable, Type

T = TypeVar("T")


class ValueType(Generic[T]):

    def __init__(self, name: str, val_type: Type[T], zero: Callable[[], T]):
        self.name: str = name
        self.type: Type[T] = val_type
        self.zero: Callable[[], T] = zero
