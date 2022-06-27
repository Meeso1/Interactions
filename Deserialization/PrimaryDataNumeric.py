from type_declarations import *
from typing import Any, List
from DeserializationError import DeserializationError
from data_types.PrimaryData import PrimaryDataNumeric
from .ValueTypes.ValueType import ValueType


def deserialize(json_object: Any, validated: bool = False) -> PrimaryDataNumeric:
    if not validated:
        if not valid(json_object):
            raise DeserializationError(PrimaryDataNumeric, json_object)

    data_type = ValueType.types[json_object["type"]]
    
    return PrimaryDataNumeric(
        val_type=data_type.type,
        initial=data_type.type.deserialize(json_object["initial"]),
        zero=data_type.zero,
        initial_derivative=data_type.type.deserialize(json_object["initial_derivative"])
        if "initial_derivative" in json_object else data_type.zero()
    )


def valid(json_object: Any) -> bool:
    """
    {
        type: string, \n
        initial: [data of corresponding type], \n
        initial_derivative: [data of corresponding type]
    }
    """
    if json_object["type"] not in ValueType.types:
        return False
    data_type = ValueType.types[json_object["type"]]
    if not data_type.checker(json_object["initial"]):
        return False
    if "initial_derivative" in json_object and not data_type.checker(json_object["initial_derivative"]):
        return False
    return True
