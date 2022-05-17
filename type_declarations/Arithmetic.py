from __future__ import annotations
from .Types import *
from typing import Protocol, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Arithmetic(Protocol[T]):
    """
    Data that can be added, negated and multiplied with floats.
    WARNING: numpy arrays that contain data that is not arithmetic pass the typecheck for some reason
    """

    def __add__(self: T, other: T) -> T:
        ...

    def __neg__(self: T) -> T:
        ...

    def __mul__(self: T, other: float | int) -> T:
        ...
