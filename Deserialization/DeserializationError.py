from typing_extensions import *
from typing import Any


class DeserializationError(Exception):

    def __init__(self, type: Type, obj: Any):
        super(DeserializationError, self).__init__(f"Not a valid {type} repr: {obj}")
