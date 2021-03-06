from __future__ import annotations
from typing import Tuple, cast, Union
from scipy import spatial                   # type: ignore
from scipy.interpolate import CubicSpline   # type: ignore
from math import inf

from data_types.TimeVariantData import CompoundData, CompoundDataDerivative
from math_primitives.FieldStepFunctions import EulerField
from math_primitives.QuarticSplineGenerator import QuarticSplineGenerator
from math_primitives.Vector import *
from math_primitives.FieldValueRepr import FieldValueRepr
from data_types.PrimaryData import PrimaryData, PrimaryDataNumeric
from type_declarations.Types import *
from numpy.typing import NDArray


class Ball:

    def __init__(self,
                 position: PrimaryDataNumeric[Vector2],
                 velocity: PrimaryDataNumeric[Vector2],
                 attributes: Dict[str, PrimaryDataNumeric] = None,
                 start_time: float = 0):

        self.position: PrimaryDataNumeric[Vector2] = position
        self.velocity: PrimaryDataNumeric[Vector2] = velocity

        # Setup attributes
        self.attributes: Dict[str, PrimaryDataNumeric] = dict() if attributes is None else attributes

        self.start_time: float = start_time
        self.time: float = self.start_time

    def update(self, new_time: float) -> None:
        # Called after Interactions update
        # Adds current velocity as current position derivative
        # Updates values of all attributes (collapse_last_val())

        if new_time <= self.time:
            raise ValueError(f"update() called with new_time that is not in the future ({new_time} <= {self.time})")
        self.time = new_time

        self.position.add_derivative((self.time, self.velocity.current))
        self.velocity.collapse_last_val()
        self.position.collapse_last_val()

        for attr in self.attributes.keys():
            self.attributes[attr].collapse_last_val()

    @staticmethod
    def make_tree(balls: List[Ball]) -> spatial.KDTree:
        # Makes KDTree containing Balls positions
        pos = np.array([b.position.current.data for b in balls])
        return spatial.KDTree(pos)

    @staticmethod
    def distance(a: Ball, b: Ball) -> float:
        return Vector.distance(a.position.current, b.position.current)

    def direction(self, other: Ball) -> Vector2:
        return (other.position.current - self.position.current).normalize()


class Field(PrimaryData):

    def __init__(self, name: str,
                 values: FieldValueRepr,
                 start_time: float = 0,
                 initial_derivative: DerivativeFunc | None = None,
                 step_func: FieldStepFunc = EulerField(),
                 create_func: Callable[[List[TimedVal[NDArray[np.float64]]]],
                                       Callable[[float], FieldValueRepr]] | None = None):
        super(Field, self).__init__()
        self.name: str = name

        self.start_time: float = start_time
        self.time: float = start_time

        # if current_derivatives is not empty, this is the time that they correspond to
        self.next_step_time: float | None = None

        self.values: FieldValueRepr = values

        # List of current derivative functions that is appended by Interactions.
        # It is later (during update) collapsed into a single function that is appended to derivatives.
        self.current_derivative_list: List[DerivativeFunc] = []

        # Field values log for plotting and analysis
        self.trace: List[TimedVal[FieldValueRepr]] = [TimedVal(
            time=self.start_time,
            val=values.copy()
        )]
        # List of data arrays of fields in trace (used for spline generation because FVRs can't be added)
        self._data_trace: List[TimedVal[NDArray[np.float64]]] = [TimedVal(
            time=self.start_time,
            val=values.get_data()
        )]
        self.derivatives: List[TimedVal[DerivativeFunc]] = [TimedVal(
            time=self.start_time,
            val=lambda x, y: 0 if initial_derivative is None else initial_derivative    # type: ignore
        )]

        self.step_func: FieldStepFunc = step_func
        self._gen = QuarticSplineGenerator(self._data_trace,
                                           np.zeros(self.values.get_data().shape),
                                           np.zeros(self.values.get_data().shape))
        self._default_create_func: Callable[[List[TimedVal[NDArray[np.float64]]]], Callable[[float], FieldValueRepr]] \
            = lambda vals: lambda t, f=self._gen(vals): type(self.values).from_values(f(t))     # type: ignore
        self.create_func: Callable[[List[TimedVal[NDArray[np.float64]]]], Callable[[float], FieldValueRepr]]
        if create_func is None:
            self.create_func = self._default_create_func
        else:
            self.create_func = create_func
        self.f: Callable[[float], FieldValueRepr] = lambda t: \
            self.trace[0].val if t == self.trace[0].time else self._raise_time_arg_error(t)     # type: ignore

    def _make_func(self) -> Callable[[float], FieldValueRepr]:
        return self.create_func(self._data_trace)

    def _raise_time_arg_error(self, time: float) -> Any:
        raise ValueError(f"time must be between start and current: {self.start_time} <= (t: {time}) <= {self.time}")

    def update(self, new_time: float) -> None:
        if new_time <= self.time:
            raise ValueError(f"update() called with new_time that is not in the future ({new_time} <= {self.time})")

        self.time = new_time

        if not self.values.const:

            # 1. Collapses current_derivative_list into one function and appends it to derivatives.
            f: DerivativeFunc = lambda x, y: sum([func(x, y) for func in self.current_derivative_list[:]])
            self.derivatives.append(TimedVal(f, self.time))

            # 2. Updates field values based on derivatives
            self.values.add(self.step_func(self.derivatives, self.time))

            # 3. Appends values to trace
            self.trace.append(TimedVal(self.values.copy(), self.time))
            self._data_trace.append(TimedVal(self.values.get_data(), self.time))
            self.f = self._make_func()

        # 4. Clears current_derivative_list
        self.current_derivative_list.clear()
        self.next_step_time = None

    def value(self, point: Vector2) -> float:
        # returns a field value at the point given.
        # Forwards arguments to vals()
        v = self.vals(np.array(point.x, dtype=np.float64), np.array(point.y, dtype=np.float64))
        if isinstance(v, float):
            return v
        elif is_array(v) and v.size == 1:
            return v[0, 0]     # This case should be redundant if vals() works correctly (but it does not :/)
        raise RuntimeError(f"vals() returned an unexpected array in response to (float, float) call: {str(v)}")

    def vals(self, points_x: CordVal, points_y: CordVal) -> Values:
        # val() for a grid of points. Forwards its arguments to values().
        return self.values(points_x, points_y)

    def val(self, time: float) -> FieldValueRepr:
        if not self.start_time <= time <= self.time:
            self._raise_time_arg_error(time)
        return self.f(time)

    def add_derivative(self, p: Tuple[float, Any]):
        if self.next_step_time is None:
            self.next_step_time = p[0]
        elif not self.next_step_time == p[0]:
            raise RuntimeError("add_derivative() for Field has to be called with current time")
        self.current_derivative_list.append(p[1])

    @property
    def range(self) -> Tuple[float, float]:
        return self.start_time, self.time

    def _get_dt(self) -> CompoundData:
        values_data = CompoundDataFieldValues(self)
        return values_data.dt

    # Analytical properties:

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


class CompoundDataFieldValues(CompoundData):

    def __init__(self, field: Field):
        super().__init__()
        self.field: Field = field

    def val(self, time: float) -> NDArray[np.float64]:
        return self.field.val(time).get_data()

    @property
    def range(self) -> Tuple[float, float]:
        return self.field.range

    def _get_dt(self) -> CompoundData:
        return CompoundDataDerivative(self)


Target: TypeAlias = Tuple[Ball, str] | Field
Source: TypeAlias = Ball | Field
Ball_Derivative_Func: TypeAlias = Callable[[Target, Source], NDArray[np.float64] | float | Vector]
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
            self.ball_select_func = self._ball_select_all_pairs

        if self.targets[0] == "Ball" and self.attribute is None:
            raise TypeError("If the Interaction targets a Ball, 'attribute' argument must be provided!")

    def update(self, new_time: float, targets: Tuple[List[Ball], Dict[str, Field]]) -> None:
        # Updates a target value - values of a Field, or a value of a chosen attribute of a Ball.
        # In (Ball, Ball) interaction a ball_select_func is used to determine which balls interact which which.
        if new_time <= self.time:
            raise ValueError(f"update() called with new_time that is not in the future ({new_time} <= {self.time})")
        self.time = new_time

        (balls, fields) = targets
        # "Self" source is used when there is no source of the interaction
        if self.targets[1] == "Self":
            if self.targets[0] == "Ball":
                for ball in balls:
                    self.update_val_ball((ball, cast(str, self.attribute)),
                                         ball,
                                         cast(Ball_Derivative_Func, self.formula))
            else:
                self.update_val_field(fields[self.targets[0]],
                                      fields[self.targets[0]],
                                      cast(Field_Derivative_Func, self.formula))

        elif self.targets == ("Ball", "Ball"):
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

    def update_val_ball(self, target: Tuple[Ball, str], source: Source, f: Ball_Derivative_Func) -> None:
        # Called in update().
        # Increments a derivative value of a corresponding attribute (target[1]) of a Ball (target[0]).
        # f is a formula of the Interaction. It takes arguments: source, target
        ball, attr_name = target
        if attr_name == "velocity":
            ball.velocity.add_derivative((self.time, cast(Vector, f(target, source))))
        else:
            ball.attributes[attr_name].add_derivative((self.time, f(target, source)))

    def update_val_field(self, target: Field, source: Source, f: Field_Derivative_Func) -> None:
        # Called in update().
        # Appends a function produced by an Interaction formula to a current_derivative_list.
        # f is a formula of the Interaction. It takes arguments: source, target
        target.add_derivative((self.time, f(target, source)))

    def _ball_select_from_tree(self, balls: List[Ball]) -> List[List[int]]:
        tree = Ball.make_tree(balls)
        return cast(List[List[int]], tree.query_ball_tree(tree, self.radius))

    @staticmethod
    def _ball_select_all_pairs(balls: List[Ball]) -> List[List[int]]:
        return [list(range(len(balls)))] * len(balls)
