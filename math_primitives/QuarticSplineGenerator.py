from dataclasses import dataclass
from typing import Generic, cast
from type_declarations.Types import *

# from scipy.optimize import minimize
# from scipy import integrate


@dataclass
class SplineSegment(Generic[N]):
    a: N
    b: N
    c: N
    d: N
    e: N
    t0: float
    t1: float

    def __call__(self, t: float) -> N:
        return self.a * (t - self.t0) ** 4 + self.b * (t - self.t0) ** 3 + self.c * (t - self.t0) ** 2 \
               + self.d * (t - self.t0) + self.e

    def dt(self, t: float) -> N:
        return 4 * self.a * (t - self.t0) ** 3 + 3 * self.b * (t - self.t0) ** 2 \
               + 2 * self.c * (t - self.t0) + self.d

    def d2t(self, t: float) -> N:
        # TODO: Fix Vector * operator to detect return type based on arguments
        return 12 * self.a * (t - self.t0) ** 2 + 6 * self.b * (t - self.t0) + 2 * self.c   # type: ignore


class QuarticSplineGenerator(Generic[N]):
    """
    Class that generates a parametrized quartic spline interpolation function from TimedVal list.
    Spline is extendable in O(1) time (for new values at the end of the list).
    Free parameter is chosen to minimize a given condition.
    """

    def __init__(self, data_list: List[TimedVal[N]], initial_derivative: N, initial_d2: N):
        # data has to contain at least 1 value
        self.data: List[TimedVal[N]] = data_list
        self.initial_derivative: N = initial_derivative
        self.initial_d2: N = initial_d2

        self.current_func_data_length: int = 1
        self.last_entry: TimedVal[N] = self.data[0]

        self.coeff: List[SplineSegment[N]] = []

        self._update_func()

    def __call__(self, values: List[TimedVal[N]]) -> Callable[[float], N]:
        if not values == self.data:
            raise ValueError("DynamicSplineGenerator can only generate spline functions from one extendable data list")
        if not self._check_data_unmodified():
            raise AttributeError("Data was modified in illegal way. Interpolation cannot be performed")

        self._update_func()

        return self.current_func

    def _check_data_unmodified(self) -> bool:
        return self.data[self.current_func_data_length - 1] == self.last_entry

    def current_func(self, t: float) -> N:
        if self.current_func_data_length == 1:
            return self.data[0].val + self.initial_derivative * (t - self.data[0].time)
        else:
            segment_index = len(self.coeff) - 1
            while segment_index > 0 and self.coeff[segment_index].t0 > t:
                segment_index -= 1
            s = self.coeff[segment_index]
            return s(t)

    def _update_func(self):
        # print(f"Spline update: {len(self.data) - self.current_func_data_length}")
        for i in range(self.current_func_data_length, len(self.data)):
            d = self.data[i - 1]
            if len(self.coeff) > 0:
                p = self.coeff[-1]
                prev_d = p.dt(d.time)
                prev_d2 = p.d2t(d.time)
            else:
                prev_d = self.initial_derivative
                prev_d2 = self.initial_d2
            self.coeff.append(self._new_coeffs(self.data[i - 1], self.data[i], prev_d, prev_d2))
        self.current_func_data_length = len(self.data)
        self.last_entry = self.data[-1]

    @staticmethod
    def _new_coeffs(p0: TimedVal[N], p1: TimedVal[N], d1: N, d2: N) -> SplineSegment:

        def a(alfa: N) -> N:
            return alfa

        def b(alfa: N) -> N:
            dt = p1.time - p0.time
            return - dt*alfa + p1.val*(1/dt**3) - 0.5*(1/dt)*d2 - (1/dt**2)*d1 - (1/dt**3)*p0.val   # type: ignore

        c = 0.5*d2
        d = d1
        e = p0.val

        x = p1.time - p0.time
        bb = p1.val * (1 / x ** 3) - 0.5 * (1 / x) * d2 - (1 / x ** 2) * d1 - (1 / x ** 3) * p0.val

        # TODO: Choose additional condition based on data

        # Least bendy
        # Usually looks ok but can be better
        # alfa_val = -(1/x)*bb - (1/(6*x**2))*c

        # Closest first derivative
        # Doesn't work :/
        # alfa_val = - (7 / 3) * (1 / x) * bb - (7 / 6) * (1 / x ** 2) * c

        # First derivative matches at end
        # Works well if first derivative changes more rapidly (+/-) or is close to 0
        # TODO: Remove cast() after fixing Vector *
        alfa_val: N = -(3/x)*bb - (2/x**2)*c - (1/x**3)*d + (1/x**4)*(p1.val - p0.val)  # type: ignore

        # First derivative matches in the middle
        # Works well for data with slowly changing first derivative
        # alfa_val = (3/x)*bb + (4/x**2)*c + (4/x**3)*d - (4/x**4)*(p1.val - p0.val)

        return SplineSegment(a(alfa_val), b(alfa_val), c, d, e, p0.time, p1.time)
