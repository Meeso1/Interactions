from typing import Tuple, Callable, Dict
import numpy as np
from BFIClasses import Ball, Field
from data_types.PrimaryData import PrimaryDataNumeric
from math_primitives.ExpSplineFieldRepr import ExpSplineFieldRepr
from math_primitives.NumericStepFunctions import Euler
from math_primitives.SplineFieldRepr import SplineFieldRepr
from math_primitives.Vector import Vector, Vector2
from type_declarations.Types import NumericStepFunc


# TODO: Fix
def make_field_vals(size: Tuple[float, float], res: Tuple[int, int],
                    func: Callable[[float, float], float]):
    cords = np.linspace(-size[0] / 2, size[0] / 2, res[0]), np.linspace(-size[1] / 2, size[1] / 2, res[1])
    values = np.zeros((cords[1].size, cords[0].size))
    for i in range(len(cords[0])):
        for j in range(len(cords[1])):
            values[j, i] = func(cords[0][i], cords[1][j])
    return values


def make_ball(start_pos: Vector2, start_v: Vector2, attrs: Dict[str, PrimaryDataNumeric] | None = None,
              start_dv: Vector2 = Vector(2), step_func: Callable[[], NumericStepFunc] = Euler) -> Ball:
    return Ball(
            position=PrimaryDataNumeric(Vector2,
                                        initial=start_pos,
                                        initial_derivative=start_v,
                                        zero=lambda: Vector(2),
                                        step_func=step_func()),
            velocity=PrimaryDataNumeric(Vector2,
                                        initial=start_v,
                                        initial_derivative=start_dv,
                                        zero=lambda: Vector(2),
                                        step_func=step_func()),
            attributes=attrs
            )


def make_field(name: str, func: Callable[[float, float], float],
               size: Tuple[float, float], res: Tuple[int, int] = (20, 20),
               positive: bool = False, const: bool = False):
    if positive:
        return Field(name, ExpSplineFieldRepr.from_values(size, make_field_vals(size, res, func), const=const))
    else:
        return Field(name, SplineFieldRepr.from_values(size, make_field_vals(size, res, func), const=const))
