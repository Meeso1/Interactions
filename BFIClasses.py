from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from scipy import spatial
from scipy.interpolate import interp2d, griddata
from math import inf

n: int = 20         # Field resolution (represented as n x n x n array)
width: float = 100  # Field width


def length(vector: np.ndarray) -> float:
    return np.linalg.norm(vector)


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = length(vector)
    if norm == 0:
        return vector * 0
    else:
        return vector * 1/norm


class Ball:

    position: np.ndarray
    velocity: np.ndarray
    attributes: Dict[str, Any]
    trace: List[Tuple[float, np.ndarray, np.ndarray]]  # position and velocity log for plotting

    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = position
        self.velocity = velocity
        self.attributes = {}
        self.trace = []
        self.time = 0

    def update(self, dt: float) -> None:
        self.position += self.velocity * dt
        self.time += dt
        self.trace.append((self.time, self.position, self.velocity))

    @staticmethod
    def make_tree(balls: List[Ball]) -> spatial.KDTree:
        pos = np.array([b.position for b in balls])
        return spatial.KDTree(pos)

    @staticmethod
    def distance(a: Ball, b: Ball) -> float:
        return np.linalg.norm(a.position - b.position)

    def direction(self, other: Ball) -> np.ndarray:
        return (other.position - self.position)*(1/Ball.distance(self, other))


class Field:

    name: str
    values: np.ndarray
    trace: List[Tuple[float, np.ndarray]]     # Field values log for plotting and analysis

    res: Tuple[int, int] = None
    size: Tuple[float, float] = None

    def __init__(self, name: str, values: np.ndarray) -> None:
        self.name = name
        self.values = values
        self.trace = []
        self.res = values.shape
        self.time = 0

    def update(self, dt: float) -> None:
        self.time += dt
        self.trace.append((self.time, self.values))

    def val(self, point: np.ndarray) -> float:
        return interp2d(np.linspace(-self.size[0]/2, self.size[0]/2, self.res[0]),
                        np.linspace(-self.size[1]/2, self.size[1]/2, self.res[1]),
                        self.values, copy=False, fill_value=0)(point[0], point[1])


class Interaction:

    Target = Union[Tuple[Ball, str], Field]
    Source = Union[Ball, Field]
    Derivative_Func = Callable[[Target, Source], Union[float, np.ndarray]]

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

    def update(self, dt: float, targets: Tuple[List[Ball], Dict[str, Field]], ball_tree: spatial.KDTree = None):
        (balls, fields) = targets
        if self.targets == ("Ball", "Ball"):
            if ball_tree is not None:
                res = ball_tree.query_ball_tree(ball_tree, self.radius)
            else:
                res = [[index for index in range(len(balls))
                        if self.radius == inf or Ball.distance(ball, balls[index]) < self.radius]
                       for ball in balls]
            for i in range(len(res)):
                for j in res[i]:
                    if j != i:
                        self.update_val((balls[i], self.attribute),
                                        balls[j],
                                        self.formula, dt)

        elif self.targets[0] == "Ball":
            for ball in balls:
                self.update_val((ball, self.attribute),
                                fields[self.targets[1]],
                                self.formula, dt)

        elif self.targets[1] == "Ball":
            for ball in balls:
                self.update_val(fields[self.targets[0]],
                                ball, self.formula, dt)

        else:
            self.update_val(fields[self.targets[0]],
                            fields[self.targets[1]],
                            self.formula, dt)

    @classmethod
    def update_val(cls, target: Target, source: Source, f: Derivative_Func, dt: float):
        if isinstance(target, Field):
            target.values += f(target, source) * dt   # TODO: Improve

        else:   # target is Ball
            if target[1] == "velocity":
                target[0].velocity += f(target, source) * dt
            else:
                target[0].attributes[target[1]] += f(target, source) * dt
