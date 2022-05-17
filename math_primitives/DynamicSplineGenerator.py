from dataclasses import dataclass
from typing import Generic
from type_declarations.Types import *


@dataclass
class DSplineSegment(Generic[N]):
    a: N
    b: N
    c: N
    d: N
    t0: float
    t1: float


class DynamicSplineGenerator(Generic[N]):
    """
    Class that generates a spline interpolation function from TimedVal list.
    Spline is extendable in O(1) time (for new values at the end of the list).
    """

    def __init__(self, data_list: List[TimedVal[N]], initial_derivative: N, initial_d2: N):
        # data has to contain at least 1 value
        self.data: List[TimedVal[N]] = data_list
        self.initial_derivative: N = initial_derivative
        self.initial_d2: N = initial_d2

        self.current_func_data_length: int = 1
        self.last_entry: TimedVal[N] = self.data[0]

        self.coeff: List[DSplineSegment[N]] = []

        self._update_func()

    def __call__(self, values: List[TimedVal[N]]) -> Callable[[float], N]:
        if not values == self.data:
            raise ValueError("DynamicSplineGenerator can only generate spline functions from one extendable data list")
        if not self._check_data_unmodified():
            raise AttributeError("Data was modified in illegal way. Interpolation cannot be performed")

        self._update_func()

        return self.current_func

    def _check_data_unmodified(self) -> bool:
        return self.data[self.current_func_data_length-1] == self.last_entry

    def current_func(self, t: float) -> N:
        if self.current_func_data_length == 1:
            return self.data[0].val + self.initial_derivative * (t - self.data[0].time)
        else:
            segment_index = len(self.coeff) - 1
            while segment_index > 0 and self.coeff[segment_index].t0 > t:
                segment_index -= 1
            s = self.coeff[segment_index]
            return s.a*t**3 + s.b*t**2 + s.c*t + s.d

    def _update_func(self):
        for i in range(self.current_func_data_length, len(self.data)):
            d = self.data[i - 1]
            if len(self.coeff) > 0:
                p = self.coeff[-1]
                prev_d = 3*p.a*d.time**2 + 2*p.b*d.time + p.c
                prev_d2 = 6*p.a*d.time + 2*p.b
            else:
                prev_d = self.initial_derivative
                prev_d2 = self.initial_d2
            self.coeff.append(self._new_coeffs(self.data[i-1], self.data[i], prev_d, prev_d2))
        self.current_func_data_length = len(self.data)
        self.last_entry = self.data[-1]

    @staticmethod
    def _new_coeffs(p0: TimedVal[N], p1: TimedVal[N], d1: N, d2: N) -> DSplineSegment:
        xp = p0.time
        xn = p1.time
        yp = p0.val
        yn = p1.val
        s = 1/(2*(xn - xp)**3)

        a = s*(- d2 * xn ** 2 + 2 * d2 * xn * xp - d2 * xp ** 2 - 2 * d1 * xn + 2 * d1 * xp - 2 * yp + 2 * yn)
        b = s*(d2 * xn ** 3 - 3 * d2 * xn * xp ** 2 + 2 * d2 * xp ** 3 + 6 * d1 * xn * xp - 6 * d1 * xp ** 2
               + 6 * xp * yp - 6 * xp * yn)
        c = s*(- 2 * d2 * xn ** 3 * xp + 3 * d2 * xn ** 2 * xp ** 2 - d2 * xp ** 4 + 2 * d1 * xn ** 3
               - 6 * d1 * xn ** 2 * xp + 4 * d1 * xp ** 3 - 6 * xp ** 2 * yp + 6 * xp ** 2 * yn)
        d = s*(d2 * xn ** 3 * xp ** 2 - 2 * d2 * xn ** 2 * xp ** 3 + d2 * xn * xp ** 4 - 2 * d1 * xn ** 3 * xp
               + 6 * d1 * xn ** 2 * xp ** 2 - 4 * d1 * xn * xp ** 3 + 2 * xn ** 3 * yp - 6 * xn ** 2 * xp * yp
               + 6 * xn * xp ** 2 * yp - 2 * xp ** 3 * yn)

        return DSplineSegment(a, b, c, d, xp, xn)
