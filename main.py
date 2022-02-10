from __future__ import annotations
import numpy as np
import typing
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import abc
import random
from scipy import spatial

n: int = 100     # Field resolution (represented as n x n x n array)


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

    Target = Union[Ball, Field]

    # static:
    interactions: List[Interaction] = []

    # instance:
    name: str
    targets: Tuple[str, str]
    formula: Callable[[Target, Target, float], None]
    radius: float

    def __init__(self,
                 name: str,
                 between: Tuple[str, str],
                 formula: Callable[[Target, Target, float], None],  # f: a,b,dt => change a
                 radius: float) -> None:
        self.name = name
        self.targets = between
        self.radius = radius
        self.formula = formula  # derivative of sth.

        Interaction.interactions.append(self)

    def update(self, dt: float):
        if self.targets == ("Ball", "Ball"):
            res = Ball.tree.query_ball_tree(Ball.tree, self.radius)
            for i in range(len(Ball.balls)):
                for j in res[i]:
                    if j != i:
                        self.formula(Ball.balls[i], Ball.balls[j], dt)

        elif self.targets[0] == "Ball":
            for ball in Ball.balls:
                self.formula(ball, Field.fields[self.targets[1]], dt)

        elif self.targets[1] == "Ball":
            for ball in Ball.balls:
                self.formula(Field.fields[self.targets[0]], ball, dt)

        else:
            self.formula(Field.fields[self.targets[0]], Field.fields[self.targets[1]], dt)

    @staticmethod
    def update_all(dt: float):
        for interaction in Interaction.interactions:
            interaction.update(dt)


def simulate(time: float, steps: int):
    dt: float = time / steps
    for i in range(n):
        Ball.update_all(dt)
        Field.update_all(dt)
        Interaction.update_all(dt)
