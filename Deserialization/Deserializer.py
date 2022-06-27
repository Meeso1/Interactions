import json
from abc import ABC, abstractmethod
from typing import Dict, Any, final, Tuple
import jsonschema

from Simulation import Simulation


class Deserializer(ABC):

    def __init__(self, schema_path: str):
        self.schema_path: str = schema_path

    @final
    def from_file(self, path: str) -> Simulation | None:
        f = open(path)
        data = json.load(f)
        f.close()

        if not self.validate(data):
            return None

        return self.deserialize(data)

    def validate(self, data: Dict[str, Any]) -> bool:
        f = open(self.schema_path, mode='r')
        schema = json.load(f)
        f.close()

        try:
            jsonschema.validate(data, schema)
        except jsonschema.exceptions.ValidationError:
            return False
        return True

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> Simulation | None:
        pass
