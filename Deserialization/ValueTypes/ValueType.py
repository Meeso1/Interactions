from type_declarations import *
from typing import TypeVar, Generic, Callable, Type, Any

T = TypeVar("T")


class ValueType(Generic[T]):

    def __init__(self, name: str, val_type: Type[T], checker: Callable[[Any], bool], zero: Callable[[], T]):
        self.name: str = name
        self.type: Type[T] = val_type
        self.checker: Callable[[Any], bool] = checker
        self.zero: Callable[[], T] = zero
