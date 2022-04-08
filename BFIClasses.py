from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from scipy import spatial
from scipy.interpolate import interp2d, griddata, RectBivariateSpline
from math import inf
from math_primitives .Vector import *

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
    attributes: Dict[str, float]
    time: float

    trace: List[Tuple[float, np.ndarray, np.ndarray]]         # Position and velocity log for plotting
    attr_derivatives: Dict[str, List[Tuple[float, float]]]    # Past derivatives used by Interactions

    def __init__(self, position: np.ndarray, velocity: np.ndarray,
                 attributes: Dict[str, Any] = None, copy_attr: bool = True) -> None:
        self.position = position
        self.velocity = velocity

        if attributes is not None:
            if copy_attr:
                self.attributes = attributes
            else:
                self.attributes = attributes.copy()
        else:
            self.attributes = dict()

        self.trace = []
        self.time = 0

        self.attr_derivatives = dict()
        for attr in self.attr_derivatives.keys():
            pass    # Initialize with zero values


    def update(self, dt: float) -> None:
        self.position += self.velocity * dt
        self.time += dt
        self.trace.append((self.time, self.position, self.velocity))
        for val in self.attr_derivatives.values():
            val.append((0, self.time))

    @staticmethod
    def make_tree(balls: List[Ball]) -> spatial.KDTree:
        pos = np.array([b.position for b in balls])
        return spatial.KDTree(pos)

    @staticmethod
    def distance(a: Ball, b: Ball) -> float:
        return np.linalg.norm(a.position - b.position)

    def direction(self, other: Ball) -> np.ndarray:
        return (other.position - self.position)*(1/Ball.distance(self, other))

    def update_attrs(self, dt: float) -> None:
        self.velocity += euler_step(self.attr_derivatives["velocity"], dt)
        for attr in self.attributes.keys():
            self.attributes[attr] += euler_step(self.attr_derivatives[attr], dt)


class Field:
    Der_func = Callable[[np.ndarray, np.ndarray], np.ndarray]

    dstep: float = 0.01

    name: str
    values: np.ndarray
    trace: List[Tuple[float, np.ndarray]]       # Field values log for plotting and analysis

    res: Tuple[int, int]
    size: Tuple[float, float]

    val_func: Callable[[np.ndarray, np.ndarray, Optional[bool]], Union[float, np.ndarray]] = None

    derivatives: List[Tuple[List[Der_func], float]]           # Used by Interactions

    def __init__(self, name: str, values: np.ndarray, size: Tuple[float, float] = None) -> None:
        self.name = name
        self.values = values
        self.trace = []
        self.res = values.shape
        self.size = size
        self.time = 0

        self.derivatives = []

        if size is not None:
            self.val_func = interp2d(np.linspace(-self.size[0]/2, self.size[0]/2, self.res[0]),
                            np.linspace(-self.size[1]/2, self.size[1]/2, self.res[1]),
                            self.values, copy=False, fill_value=0)

    def make_val_func(self) -> Callable[[np.ndarray, np.ndarray, Optional[bool]], float]:
        return interp2d(np.linspace(-self.size[0] / 2, self.size[0] / 2, self.res[0]),
                        np.linspace(-self.size[1] / 2, self.size[1] / 2, self.res[1]),
                        self.values, copy=False, kind='cubic')

    def update(self, dt: float) -> None:
        self.time += dt
        self.trace.append((self.time, self.values))
        self.val_func = self.make_val_func()
        self.derivatives.append(([], self.time))

    def update_val(self, dt: float) -> None:
        d_list = Interaction.field_der_funcs(self, 1)
        d_vals = [(func(self.values_cords()[0], self.values_cords()[1]), 0) for func in d_list]
        self.values += euler_step(d_vals, dt)

    def values_cords(self):
        return np.linspace(-self.size[0] / 2, self.size[0] / 2, self.res[0]), \
               np.linspace(-self.size[1] / 2, self.size[1] / 2, self.res[1])

    # analytical properties:
    def val(self, point: np.ndarray) -> float:
        if self.val_func is None:
            self.val_func = self.make_val_func()
        return self.val_func(point[0], point[1])

    def vals(self, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        return self.val_func(points_x, points_y, assume_sorted=True)

    def dx(self, point: np.ndarray) -> float:
        h = self.size[0]*self.dstep
        return ((self.val(point + [h, 0]) - self.val(point - [h, 0])) / (2 * h))[0]

    def dx_mat(self, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        h = self.size[0] * self.dstep
        return (self.vals(points_x + h, points_y) - self.vals(points_x - h, points_y)) / (2 * h)

    def dy(self, point: np.ndarray) -> float:
        h = self.size[1]*self.dstep
        return ((self.val(point + [0, h]) - self.val(point - [0, h])) / (2 * h))[0]

    def dy_mat(self, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        h = self.size[1] * self.dstep
        return (self.vals(points_x, points_y + h) - self.vals(points_x, points_y + h)) / (2 * h)

    def dx2(self, point: np.ndarray) -> float:
        h = self.size[0]*self.dstep
        return (self.val(point + [h, 0]) - 2 * self.val(point) + self.val(point - [h, 0])) / (h ** 2)

    def dx2_mat(self, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        h = self.size[0] * self.dstep
        return (self.vals(points_x + h, points_y) - 2 * self.vals(points_x, points_y)
                + self.vals(points_x - h, points_y)) / (h ** 2)

    def dy2(self, point: np.ndarray) -> float:
        h = self.size[1]*self.dstep
        return (self.val(point + [0, h]) - 2 * self.val(point) + self.val(point - [0, h])) / (h ** 2)

    def dy2_mat(self, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        h = self.size[1] * self.dstep
        return (self.vals(points_x, points_y + h) - 2 * self.vals(points_x, points_y)
                + self.vals(points_x, points_y - h)) / (h ** 2)

    def gradient(self, point: np.ndarray) -> np.ndarray:
        return np.array([self.dx(point), self.dy(point)])


class Interaction:

    time: float = 0

    Target = Union[Tuple[Ball, str], Field]
    Source = Union[Ball, Field]

    Ball_Derivative_Func = Callable[[Target, Source], np.ndarray]
    Field_Derivative_Func = Callable[[Target, Source], Field.Der_func]

    name: str
    targets: Tuple[str, str]
    formula: Union[Ball_Derivative_Func, Field_Derivative_Func]
    attribute: str
    radius: float

    def __init__(self,
                 name: str,
                 between: Tuple[str, str],
                 formula: Union[Ball_Derivative_Func, Field_Derivative_Func],  # f: a,b => da/dt
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
                        self.update_val_ball((balls[i], self.attribute),
                                             balls[j],
                                             self.formula, dt)

        elif self.targets[0] == "Ball":
            for ball in balls:
                self.update_val_ball((ball, self.attribute),
                                     fields[self.targets[1]],
                                     self.formula, dt)

        elif self.targets[1] == "Ball":
            for ball in balls:
                self.update_val_field(fields[self.targets[0]],
                                      ball, self.formula, dt)

        else:
            self.update_val_field(fields[self.targets[0]],
                                  fields[self.targets[1]],
                                  self.formula, dt)

    @staticmethod
    def update_val_ball(target: Tuple[Ball, str], source: Source, f: Ball_Derivative_Func, dt: float) -> None:
        ball, attr_name = target
        if attr_name == "velocity":
            ball.attr_derivatives["velocity"][-1] += f(target, source)
        else:
            ball.attr_derivatives[attr_name][-1] += f(target, source)

    @staticmethod
    def update_val_field(target: Field, source: Source, f: Field_Derivative_Func, dt: float) -> None:
        target.derivatives[-1][0].append(f(target, source))

    @staticmethod
    def field_der_funcs(field: Field, steps_back: int) -> List[Field.Der_func]:
        output = []
        for i in range(steps_back):
            output.insert(0, lambda x, y: sum([func(x, y) for func in field.derivatives[-i-1][0]]))
        return output


def euler_step(d_vals: List[Tuple[np.ndarray, float]], dt: float) -> np.ndarray:
    return d_vals[-1][0]*dt


def method2_step(d_vals: List[Tuple[np.ndarray, float]], dt: float) -> np.ndarray:
    return (3/2*d_vals[-1][0] - 1/2*d_vals[-2][0]) * dt

