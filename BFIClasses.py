from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from scipy import spatial
from math import inf

n: int = 20         # Field resolution (represented as n x n x n array)
width: float = 100  # Field width


class Ball:

    # static:
    balls: List[Ball] = []
    tree: spatial.KDTree = None

    # instance:
    position: np.ndarray
    velocity: np.ndarray
    attributes: Dict[str, Any]
    trace: List[Tuple[np.ndarray, np.ndarray]]  # position and velocity log for plotting

    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = position
        self.velocity = velocity
        self.attributes = {}
        self.trace = []

        Ball.balls.append(self)

    def update(self, dt: float) -> None:
        self.position += self.velocity * dt
        self.trace.append((self.position, self.velocity))

    @staticmethod
    def update_all(dt: float) -> None:
        for ball in Ball.balls:
            ball.update(dt)

    @staticmethod
    def make_tree() -> None:
        pos = np.array([b.position for b in Ball.balls])
        Ball.tree = spatial.KDTree(pos)


class Field:

    # static:
    fields: Dict[str, Field] = {}

    # instance:
    name: str
    values: np.ndarray
    trace: List[np.ndarray]     # Field values log for plotting and analysis

    def __init__(self, name: str, values: np.ndarray) -> None:
        self.name = name
        self.values = values

        Field.fields[self.name] = self

    def update(self, dt: float):
        self.trace.append(self.values)

    @staticmethod
    def update_all(dt: float):
        for (key, value) in Field.fields:
            value.update(dt)


class Interaction:

    Target = Union[Tuple[Ball, str], Field]
    Source = Union[Ball, Field]
    Derivative_Func = Callable[[Target, Source], Union[float, np.ndarray]]

    # static:
    interactions: List[Interaction] = []

    # instance:
    name: str
    targets: Tuple[str, str]
    formula: Derivative_Func
    attribute: str
    radius: float

    def __init__(self,
                 name: str,
                 between: Tuple[str, str],
                 formula: Derivative_Func,  # f: a,b => da/dt
                 radius: Optional[float] = inf,
                 attribute: Optional[str] = None) -> None:
        self.name = name
        self.targets = between
        self.radius = radius
        self.attribute = attribute
        self.formula = formula  # derivative of sth.

        Interaction.interactions.append(self)

    def update(self, dt: float):
        if self.targets == ("Ball", "Ball"):
            res = Ball.tree.query_ball_tree(Ball.tree, self.radius)
            for i in range(len(Ball.balls)):
                for j in res[i]:
                    if j != i:
                        self.update_val((Ball.balls[i], self.attribute),
                                        Ball.balls[j],
                                        self.formula, dt)

        elif self.targets[0] == "Ball":
            for ball in Ball.balls:
                self.update_val((ball, self.attribute),
                                Field.fields[self.targets[1]],
                                self.formula, dt)

        elif self.targets[1] == "Ball":
            for ball in Ball.balls:
                self.update_val(Field.fields[self.targets[0]],
                                ball, self.formula, dt)

        else:
            self.update_val(Field.fields[self.targets[0]],
                            Field.fields[self.targets[1]],
                            self.formula, dt)

    @staticmethod
    def update_all(dt: float):
        for interaction in Interaction.interactions:
            interaction.update(dt)

    @classmethod
    def update_val(cls, target: Target, source: Source, f: Derivative_Func, dt: float):
        if isinstance(target, Field):
            target.values += f(target, source) * dt   # TODO: Improve

        else:   # target is Ball
            if target[1] == "velocity":
                target[0].velocity += f(target, source) * dt
            else:
                target[0].attributes[target[1]] += f(target, source) * dt

