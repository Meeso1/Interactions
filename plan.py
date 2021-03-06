from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any, TypeAlias
from scipy import spatial
from math_primitives import FieldValueRepr
from math_primitives.FieldValueRepr import CordVal, Values
from numpy.typing import NDArray


class Vector2:
    # Includes ==, +=, +, -
    # Includes (x, y) constructor, () constructor, and np.ndarray constructor

    shape: Tuple[int, int] = (2, 1)     # const

    x: float
    y: float


class Ball:
    position: Vector2
    velocity: Vector2
    time: float

    attributes: Dict[str, Any]
    # Initialized with starting parameters that determine the type
    # To get zero value, use "type(prev)()"

    trace: List[Tuple[float, Vector2, Vector2]]
    # Position and velocity log for plotting

    attr_derivatives: Dict[str, List[Tuple[Any, float]]]
    # Past derivatives used by Interactions
    # New entries created before Interactions update

    def init_attr_derivatives(self, time: float):
        # Initializes new entries in every attr_derivatives list
        # New values are default - type(prev_val)()
        pass

    def update(self, dt: float) -> None:
        # Called after Interactions update
        # Updates attrs based on attr_derivatives, using some method
        # Updates velocity and position
        # Step number is determined based on length of attr_derivatives lists, or trace
        pass

    @staticmethod
    def make_tree(balls: List[Ball]) -> spatial.KDTree:
        # Makes KDTree
        pass


DerivativeFunc: TypeAlias = Callable[[CordVal, CordVal], Values]
# Type of field derivative function.
# For arguments x = [x1, x2, x3, ...] and y = [y1, y2, y3, ...]
# returns an array of derivatives (d/dt) at points x*y (vector product)


class Field:
    name: str
    values: FieldValueRepr
    trace: List[Tuple[float, NDArray[np.float64]]]  # Field values log for plotting and analysis

    derivatives: List[Tuple[DerivativeFunc, float]]
    # List of tuples containing timestamps and derivative functions at that step.

    current_derivative_list: List[DerivativeFunc]
    # List of current derivative functions that is appended by Interactions.
    # It is later (during update) collapsed into a single function that is appended to derivatives.

    def update(self, dt: float) -> None:
        # 1. Collapses current_derivative_list into one function and appends it to derivatives.
        # 2. Updates field values based on derivatives (values += func)
        # 3. Appends values to trace
        # 4. Clears current_derivative_list
        pass

    def val(self, point: Vector2) -> float:
        # returns a field value at the point given.
        # Forwards arguments to vals()
        pass

    def vals(self, points_x: CordVal, points_y: CordVal) -> Values:
        # val() for a grid of points. Forwards its arguments to values().
        pass

    # Analytical properties:
    # dx, dy    - using an external derivative() function
    # dt        - interpolated from derivatives list
    # dx2, dy2  - with external derivative2() function
    # gradient  - with dx and dy
    # dir       - directional derivative
    # div, rot  - divergence and rotation
    # angle     - slope of the field in given direction


Target: TypeAlias = Tuple[Ball, str] | Field
# Type that represents the target of the interaction - the thing that will be changed by it.
# If it is a Field, Interaction produces the derivative of its values.
# If it is a Ball, second item in a tuple specifies the name of the attribute to be changed,
# or "velocity" (position shouldn't be changed directly, except in Collisions).
Source: TypeAlias = Ball | Field
# Specifies the source of the interaction. Used in formula.
Ball_Derivative_Func: TypeAlias = Callable[[Target, Source], Any]
Field_Derivative_Func: TypeAlias = Callable[[Target, Source], DerivativeFunc]
Formula: TypeAlias = Ball_Derivative_Func | Field_Derivative_Func


class Interaction:
    time: float = 0
    name: str
    attribute: str
    targets: Tuple[str, str]
    # Specifies what the Interaction exists between. "Ball" specifies any Ball, field name specifies field.
    # If targets[0] is "Ball", attribute specifies which attribute of the Ball will be changed.
    # Interaction formula will be called with an argument (ball, attribute) for every ball in simulation.

    formula: Formula
    # A formula that generates a derivative.

    radius: float
    # Radius of the interaction between Balls. KDTree is used to select all pairs of Balls that are close enough.

    ball_select_func: Callable[[List[Ball]], List[List[int]]]
    # A function that selects Balls that are "close enough" to interact (alternative to KDTree query).
    # If None, KDTree query is used. None by default.

    def update(self, dt: float, targets: Tuple[List[Ball], Dict[str, Field]]):
        # Updates a target value - values of a Field, or a value of a chosen attribute of a Ball.
        # targets = (balls, fields)
        # In (Ball, Ball) interaction a ball_select_func is used to determine which balls interact which which.
        pass

    @staticmethod
    def update_val_ball(target: Tuple[Ball, str], source: Source, f: Ball_Derivative_Func) -> None:
        # Called in update().
        # Increments a derivative value of a corresponding attribute (target[1]) of a Ball (target[0]).
        # f is a formula of the Interaction. It takes arguments: source, target
        pass

    @staticmethod
    def update_val_field(target: Field, source: Source, f: Field_Derivative_Func) -> None:
        # Called in update().
        # Appends a function produced by an Interaction formula to a current_derivative_list.
        # f is a formula of the Interaction. It takes arguments: source, target
        pass
