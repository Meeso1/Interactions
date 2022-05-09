from type_declarations.Types import *
from typing import List


# Required steps: 2
def euler(derivatives: List[TimedVal[N]], time: float | None = None) -> N:

    if time is None:
        time = derivatives[-1].time

    curr = derivatives[1]
    prev = derivatives[0]
    for i in range(len(derivatives) - 1):
        if derivatives[-i-2].time < time <= derivatives[-i-1].time:
            curr = derivatives[-i-1]
            prev = derivatives[-i-2]
            break

    return (curr.time - prev.time) * curr.val   # type: ignore
