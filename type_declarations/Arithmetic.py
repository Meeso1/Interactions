from __future__ import annotations
from typing import Protocol, runtime_checkable, TypeVar

T = TypeVar("T")


@runtime_checkable
class Arithmetic(Protocol):
    """
    Data that can be added, negated and multiplied with floats.
    WARNING: numpy arrays that contain data that is not arithmetic pass the typecheck for some reason
    """

    def __add__(self: T, other: T) -> T:
        ...

    def __sub__(self: T, other: T) -> T:
        ...

    def __neg__(self: T) -> T:
        ...

    def __mul__(self: T, other: float) -> T:
        ...

    def __rmul__(self: T, other: float) -> T:
        ...
