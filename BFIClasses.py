from __future__ import annotations
from typing import List, Tuple, Optional, Callable, cast, Union
from scipy import spatial   # type: ignore
from math import inf
from math_primitives.Vector import *
from math_primitives.FieldValueRepr import CordVal, Values, FieldValueRepr
from numpy.typing import NDArray


class Ball:

    position: Vector2
    velocity: Vector2
    attributes: Dict[str, Any]
    time: float

    trace: List[Tuple[float, Vector2, Vector2]]                 # Position and velocity log for plotting
    attr_derivatives: Dict[str, List[List[Any]]]        # Past derivatives used by Interactions

    def __init__(self, position: Vector2, velocity: Vector2,
                 attributes: Dict[str, Any] = None, copy_attr: bool = True) -> None:
        self.position: Vector2 = position
        self.velocity: Vector2 = velocity

        # Setup attributes
        self.attributes: Dict[str, Any]
        if attributes is not None:
            if copy_attr:
                self.attributes = attributes
            else:
                self.attributes = attributes.copy()
        else:
            self.attributes = dict()

        self.trace: List[Tuple[float, Vector2, Vector2]] = []
        self.time: float = 0

        self.attr_derivatives: Dict[str, List[List[Any]]] = {}
        for attr in self.attributes.keys():
            self.attr_derivatives[attr] = []

    def init_attr_derivatives(self, time: float):
        # Initializes new entries in every attr_derivatives list
        # New values are default - type(prev_val)()
        self.time = time
        for key, val in self.attr_derivatives.items():
            if key == "velocity":
                val.append([time, Vector(2)])
                continue

            a = self.attributes[key]
            if isinstance(a, Vector):
                zero_val = Vector(a.n)
            else:
                zero_val = type(a)()
            val.append([time, zero_val])

    def update(self, dt: float) -> None:
        # Called after Interactions update
        # Updates attrs based on attr_derivatives, using some method
        # Updates velocity and position
        # Step number is determined based on length of trace

        self.trace.append((self.time, self.position, self.velocity))
        step = len(self.trace)  # In the future will be used to switch to multistep methods

        self.position += euler_step([[t, v] for (t, pos, v) in self.trace[-1:]], dt)
        self.velocity += euler_step(self.attr_derivatives["velocity"], dt)
        for attr in self.attributes.keys():
            self.attributes[attr] += euler_step(self.attr_derivatives[attr], dt)

    @staticmethod
    def make_tree(balls: List[Ball]) -> spatial.KDTree:
        # Makes KDTree containing Balls positions
        pos = np.array([b.position.data for b in balls])
        return spatial.KDTree(pos)

    @staticmethod
    def distance(a: Ball, b: Ball) -> float:
        return Vector.distance(a.position, b.position)

    def direction(self, other: Ball) -> Vector2:
        return (other.position - self.position).normalize()


DerivativeFunc: TypeAlias = Callable[[CordVal, CordVal], Values]
# Type of field derivative function.
# For arguments x = [x1, x2, x3, ...] and y = [y1, y2, y3, ...]
# returns an array of derivatives (d/dt) at points x*y (vector product)


class Field:
    name: str
    time: float
    values: FieldValueRepr

    trace: List[Tuple[float, NDArray[np.float64]]]
    # Field values log for plotting and analysis

    derivatives: List[Tuple[float, DerivativeFunc]]
    # List of tuples containing timestamps and derivative functions at that step.

    current_derivative_list: List[DerivativeFunc]
    # List of current derivative functions that is appended by Interactions.
    # It is later (during update) collapsed into a single function that is appended to derivatives.

    def __init__(self, name: str, values: FieldValueRepr) -> None:
        self.name: str = name
        self.values: FieldValueRepr = values
        self.trace: List[Tuple[float, NDArray[np.float64]]] = []
        self.time: float = 0

        self.derivatives: List[Tuple[float, DerivativeFunc]] = []
        self.current_derivative_list: List[DerivativeFunc] = []

    def update(self, dt: float) -> None:
        self.time += dt

        # 1. Collapses current_derivative_list into one function and appends it to derivatives.
        f: DerivativeFunc = lambda x, y: sum([func(x, y) for func in self.current_derivative_list[:]])
        self.derivatives.append((self.time, f))

        # 2. Updates field values based on derivatives
        self.values += euler_step_field(self.derivatives, dt)

        # 3. Appends values to trace
        self.trace.append((self.time, self.values.get_data()))

        # 4. Clears current_derivative_list
        self.current_derivative_list.clear()

    def val(self, point: Vector2) -> float:
        # returns a field value at the point given.
        # Forwards arguments to vals()
        v = self.vals(np.array(point.x, dtype=np.float64), np.array(point.y, dtype=np.float64))
        if isinstance(v, float):
            return v
        elif is_array(v) and v.size == 1:
            return v[0, 0]     # This case is currently redundant
        raise RuntimeError(f"vals() returned an unexpected array in response to (float, float) call: {str(v)}")

    def vals(self, points_x: CordVal, points_y: CordVal) -> Values:
        # val() for a grid of points. Forwards its arguments to values().
        return self.values(points_x, points_y)

    # Analytical properties:
    # dx, dy    - using an external derivative() function
    # TODO: dt  - interpolated from derivatives list
    # dx2, dy2  - with external derivative2() function
    # gradient  - with dx and dy
    # dir       - directional derivative
    # div, rot  - divergence and rotation (vector fields only!)
    # angle     - slope of the field in given direction

    def dx(self, x: CordVal, y: CordVal) -> Values:
        return self.values.dx(x, y)

    def dy(self, x: CordVal, y: CordVal) -> Values:
        return self.values.dy(x, y)

    def dx2(self, x: CordVal, y: CordVal) -> Values:
        return self.values.dx2(x, y)

    def dy2(self, x: CordVal, y: CordVal) -> Values:
        return self.values.dy2(x, y)

    def gradient(self, x: float, y: float) -> Vector2:
        return Vector2([self.dx(x, y), self.dy(x, y)])

    def dir(self, x: CordVal, y: CordVal, v: Vector2) -> Values:
        return self.dx(x, y)*v.x + self.dy(x, y)*v.y

    def angle(self, x: CordVal, y: CordVal, v: Vector2) -> Values:
        return np.arctan(self.dir(x, y, v))


Target: TypeAlias = Tuple[Ball, str] | Field
Source: TypeAlias = Ball | Field
Ball_Derivative_Func: TypeAlias = Callable[[Target, Source], np.ndarray]
Field_Derivative_Func: TypeAlias = Callable[[Target, Source], DerivativeFunc]
Formula: TypeAlias = Union[Ball_Derivative_Func, Field_Derivative_Func]


class Interaction:
    time: float
    name: str
    attribute: str | None
    targets: Tuple[str, str]
    # Specifies what the Interaction exists between. "Ball" specifies any Ball, field name specifies field.
    # If targets[0] is "Ball", attribute specifies which attribute of the Ball will be changed.
    # Interaction formula will be called with an argument (ball, attribute) for every ball in simulation.

    formula: Formula
    # A formula that generates a derivative.

    radius: float
    # Radius of the interaction between Balls. KDTree is used to select all pairs of Balls that are close enough.

    # ball_select_func: Callable[[List[Ball]], List[List[int]]]
    # A function that selects Balls that are "close enough" to interact (alternative to KDTree query).
    # If None, KDTree query is used. None by default.

    def __init__(self,
                 name: str,
                 between: Tuple[str, str],
                 formula: Formula,  # f: a,b => da/dt
                 radius: float = inf,
                 attribute: Optional[str] = None,
                 ball_select_func: Optional[Callable[[List[Ball]], List[List[int]]]] = None):
        self.name: str = name
        self.time: float = 0
        self.targets: Tuple[str, str] = between
        self.radius: float = radius
        self.attribute: str | None = attribute
        self.formula: Formula = formula  # derivative of sth.

        self.ball_select_func: Callable[[List[Ball]], List[List[int]]]
        if ball_select_func is not None:
            self.ball_select_func = ball_select_func
        else:
            f: Callable[[List[Ball]], List[List[int]]] = lambda a: self._ball_select_from_tree(a)
            self.ball_select_func = f

        if self.targets[0] == "Ball" and self.attribute is None:
            raise TypeError("If the Interaction targets a Ball, 'attribute' argument must be provided!")

    def update(self, dt: float, targets: Tuple[List[Ball], Dict[str, Field]]) -> None:
        # Updates a target value - values of a Field, or a value of a chosen attribute of a Ball.
        # In (Ball, Ball) interaction a ball_select_func is used to determine which balls interact which which.
        (balls, fields) = targets
        if self.targets == ("Ball", "Ball"):
            res = self.ball_select_func(balls)
            for i in range(len(res)):
                for j in res[i]:
                    if j != i:
                        self.update_val_ball((balls[i], cast(str, self.attribute)),
                                             balls[j],
                                             cast(Ball_Derivative_Func, self.formula))
        elif self.targets[0] == "Ball":
            for ball in balls:
                self.update_val_ball((ball, cast(str, self.attribute)),
                                     fields[self.targets[1]],
                                     cast(Ball_Derivative_Func, self.formula))
        elif self.targets[1] == "Ball":
            for ball in balls:
                self.update_val_field(fields[self.targets[0]],
                                      ball,
                                      cast(Field_Derivative_Func, self.formula))
        else:
            self.update_val_field(fields[self.targets[0]],
                                  fields[self.targets[1]],
                                  cast(Field_Derivative_Func, self.formula))

    @staticmethod
    def update_val_ball(target: Tuple[Ball, str], source: Source, f: Ball_Derivative_Func) -> None:
        # Called in update().
        # Increments a derivative value of a corresponding attribute (target[1]) of a Ball (target[0]).
        # f is a formula of the Interaction. It takes arguments: source, target
        ball, attr_name = target
        if attr_name == "velocity":
            ball.attr_derivatives["velocity"][-1][1] += f(target, source)
        else:
            ball.attr_derivatives[attr_name][-1][1] += f(target, source)

    @staticmethod
    def update_val_field(target: Field, source: Source, f: Field_Derivative_Func) -> None:
        # Called in update().
        # Appends a function produced by an Interaction formula to a current_derivative_list.
        # f is a formula of the Interaction. It takes arguments: source, target
        target.values += f(target, source)

    def _ball_select_from_tree(self, balls: List[Ball]) -> List[List[int]]:
        tree = Ball.make_tree(balls)
        return cast(List[List[int]], tree.query_ball_tree(tree, self.radius))


def euler_step(d_vals: List[List[Any]], dt: float) -> Any:
    return d_vals[-1][1]*dt


def euler_step_field(d_vals: List[Tuple[float, DerivativeFunc]], dt: float) -> DerivativeFunc:
    res: DerivativeFunc = lambda x, y: d_vals[-1][1](x, y)*dt
    return res


def method2_step(d_vals: List[Tuple[np.ndarray, float]], dt: float) -> np.ndarray:
    return (3/2*d_vals[-1][0] - 1/2*d_vals[-2][0]) * dt
