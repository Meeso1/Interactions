import json
from abc import ABC, abstractmethod
from typing import Dict, Any, final

from Simulation import Simulation


# TODO: Validation with schema
class Deserializer(ABC):

    @final
    def from_file(self, path: str) -> Simulation:
        f = open(path)
        data = json.load(f)
        f.close()
        return self.deserialize(data)

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> Simulation:
        pass
