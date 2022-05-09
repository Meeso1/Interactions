from dataclasses import dataclass
from typing import Generic, TypeVar

A = TypeVar("A")


@dataclass
class TimedVal(Generic[A]):
    val: A
    time: float
