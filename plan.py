from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from scipy import spatial


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


class Field:
    Der_func = Callable[[np.ndarray, np.ndarray], np.ndarray]
    # Type of field derivative function.
    # For arguments x = [x1, x2, x3, ...] and y = [y1, y2, y3, ...]
    # returns an array of derivatives (d/dt) at points x*y (vector product)

    name: str
    values: np.ndarray
    trace: List[Tuple[float, np.ndarray]]  # Field values log for plotting and analysis

    res: Tuple[int, int]
    size: Tuple[float, float]

    val_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    # Function that returns field value at given point (interpolated).
    # For arguments x = [x1, x2, x3, ...] and y = [y1, y2, y3, ...]
    # returns an array of values at points x*y (vector product).

    derivatives: List[Tuple[Der_func, float]]
    # List of tuples containing timestamps and derivative functions at that step.

    current_derivative_list: List[Der_func]
    # List of current derivative functions that is appended by Interactions.
    # It is later (during update) collapsed into a single function that is appended to derivatives.

    def make_val_func(self) -> Callable[[np.ndarray, np.ndarray], float]:
        # Returns a value to be assigned to val_func.
        # Called at the end of update.
        pass

    def update(self, dt: float) -> None:
        # 1. Collapses current_derivative_list into one function and appends it to derivatives.
        # 2. Updates field values based on derivatives
        # 3. Appends values to trace
        # 4. Clears current_derivative_list
        pass

    def values_cords(self) -> Tuple[np.ndarray, np.ndarray]:
        # Returns coordinates that correspond to points at which field value is stored.
        # The value returned should be compatible with a format used by val_func.
        # Namely, it should be in form of 2 np.ndarrays with x and y coordinates.
        pass

    def val(self, point: Vector2) -> float:
        # returns a field value at the point given.
        pass

    def vals(self, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        # val() for a grid of points. Forwards its arguments to val_func.
        pass

    # Analytical properties:
    # dx, dy    - using an external derivative() function
    # dt        - interpolated from derivatives list
    # dx2, dy2  - with external derivative2() function
    # gradient  - with dx and dy
    # dir       - directional derivative
    # div, rot  - divergence and rotation
    # angle     - slope of the field in given direction


class Interaction:
    Target = Union[Tuple[Ball, str], Field]
    # Type that represents the target of the interaction - the thing that will be changed by it.
    # If it is a Field, Interaction produces the derivative of its values.
    # If it is a Ball, second item in a tuple specifies the name of the attribute to be changed,
    # or "velocity" (position shouldn't be changed directly, except in Collisions).

    Source = Union[Ball, Field]
    # Specifies the source of the interaction. Used in formula.

    time: float = 0

    Ball_Derivative_Func = Callable[[Target, Source], np.ndarray]
    Field_Derivative_Func = Callable[[Target, Source], Field.Der_func]

    name: str
    attribute: str
    targets: Tuple[str, str]
    # Specifies what the Interaction exists between. "Ball" specifies any Ball, field name specifies field.
    # If targets[0] is "Ball", attribute specifies which attribute of the Ball will be changed.
    # Interaction formula will be called with an argument (ball, attribute) for every ball in simulation.

    formula: Union[Ball_Derivative_Func, Field_Derivative_Func]
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
    def update_val_field(target: Field, source: Source, f: Field_Derivative_Func, dt: float) -> None:
        # Called in update().
        # Appends a function produced by an Interaction formula to a current_derivative_list.
        # f is a formula of the Interaction. It takes arguments: source, target
        pass
