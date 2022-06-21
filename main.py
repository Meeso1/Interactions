from __future__ import annotations
from Simulation import *
from utility import *
from data_types.ExpSplineFieldRepr import ExpSplineFieldRepr
from math_primitives.NumericStepFunctions import Euler
from data_types.SplineFieldRepr import SplineFieldRepr
from math_primitives.Vector import Vector


def test0():
    sim = Simulation(
        start_balls=[
            make_ball(Vector([15, -100]), Vector([0, 150])),
            make_ball(Vector([-15, 100]), Vector([0, -180])),
            make_ball(Vector([-100, 75]), Vector([162, 0]))
        ],
        fields={},
        interactions=[],
        total_time=15, steps_per_second=20, field_size=(1000, 600))

    sim_display = SimulationDisplay(sim, 50, 8, scale=1,
                                    background_field=None,
                                    background_field_res=(50, 50),
                                    background_field_range=(-0.05, 0.68))
    sim_display.display_sim()


def test1():
    repellent_values = np.ones((20, 20), dtype=float)
    x_lim = repellent_values.shape[0]
    y_lim = repellent_values.shape[1]
    field_center = (0, 5)
    for i in range(x_lim):
        for j in range(y_lim):
            repellent_values[i, j] = 10 * 1 / \
                                     (25 * np.linalg.norm((i - x_lim / 2 + 0.5 - field_center[0],
                                                           j - y_lim / 2 + 0.5 - field_center[1])) ** 2 + 1)

    print((repellent_values.min(initial=inf),
           repellent_values.sum() / repellent_values.size,
           repellent_values.max(initial=-inf)))

    field_repr = SplineFieldRepr.from_values((1000, 600), repellent_values)

    sim = Simulation(
        start_balls=[
            make_ball(Vector([15, -100]), Vector([0, 150])),
            make_ball(Vector([-15, 100]), Vector([0, -180])),
            make_ball(Vector([-100, 75]), Vector([162, 0]))
        ],
        fields={"repellent": Field("repellent", field_repr)},
        interactions=[Interaction("Repulsion", ("Ball", "repellent"),
                      lambda target, repellent: 1000 * target[0].position.current.normalize() * repellent.value(
                         target[0].position.current),
                     attribute="velocity"),
         Interaction("Diffusion", ("repellent", "repellent"),
                     lambda _, rep: lambda x, y: 250 * (rep.dx2(x, y) + rep.dy2(x, y)))],
        total_time=15, steps_per_second=100, field_size=(1000, 600))

    sim_display = SimulationDisplay(sim, 50, 8, scale=1,
                                    background_field="repellent",
                                    background_field_res=(50, 50),
                                    background_field_range=(-0.05, 0.68))
    sim_display.display_sim()


def test2():
    height_values = np.ones((20, 20), dtype=float)
    x_lim = height_values.shape[0]
    y_lim = height_values.shape[1]
    peaks = [(0, 0), (-2, 3), (1.5, 7)]
    heights = [10000, 16000, 8000]
    for i in range(x_lim):
        for j in range(y_lim):
            val = 0
            for peak, height in zip(peaks, heights):
                val += height / (3 * np.linalg.norm((i - x_lim / 2 + 0.5 - peak[0],
                                                     j - y_lim / 2 + 0.5 - peak[1])) ** 2 + 1)
            height_values[i, j] = val

    print((height_values.min(initial=inf),
           height_values.sum() / height_values.size,
           height_values.max(initial=-inf)))

    sim = Simulation(
        start_balls=[
                make_ball(Vector([15, -100]), Vector([0, 150])),
                make_ball(Vector([-15, 100]), Vector([0, -180])),
                make_ball(Vector([-100, 75]), Vector([162, 0])),
            ],
        fields={"height": Field("height", ExpSplineFieldRepr.from_values((1000, 600), height_values, const=True))},
        interactions=[Interaction("Gravity", ("Ball", "height"),
                                  lambda target, h: -20 * h.gradient(target[0].position.current.x,
                                                                     target[0].position.current.y),
                                  attribute="velocity")],
        total_time=5, steps_per_second=50, field_size=(1000, 600))

    sim_display = SimulationDisplay(sim, 50, 8, scale=1,
                                    background_field="height",
                                    background_field_res=(60, 60),
                                    background_field_range=(0, 7000))
    sim_display.display_sim()


def test3():
    temp_field = make_field_vals((1000, 600), (50, 50), lambda x, y: 10000 / (x ** 2 + y ** 2 + 1))

    print((temp_field.min(initial=inf), temp_field.sum() / temp_field.size, temp_field.max(initial=-inf)))

    sim = Simulation(
        start_balls=[
            make_ball(Vector([15, -100]), Vector([0, 150])),
            make_ball(Vector([-15, 100]), Vector([0, -180])),
            make_ball(Vector([-100, 75]), Vector([162, 0])),
        ],
        fields={"temperature": make_field("temperature", size=(1000, 600), res=(50, 50),
                                          func=lambda x, y: 10000 / (x ** 2 + y ** 2 + 1))},
        interactions=[Interaction("Too hot!", ("Ball", "temperature"),
                      lambda target, temp: -10000 * temp.gradient(target[0].position.current.x,
                                                                  target[0].position.current.y)
                                           - 5000 * temp.dir(target[0].position.current.x,
                                                             target[0].position.current.y,
                                                             target[0].velocity.current.normalize())
                      * target[0].velocity.current.normalize(),
                     attribute="velocity")],
        total_time=15, steps_per_second=150, field_size=(1000, 600))

    test_point = Vector([10, 10])
    print(sim.fields["temperature"].value(test_point))
    print(sim.fields["temperature"].gradient(test_point.x, test_point.y))

    disp = SimulationDisplay(sim, 50, 8, scale=1,
                             background_field="temperature",
                             background_field_res=(60, 60),
                             background_field_range=(0, 30))
    disp.display_sim()


def test4(step_func: Callable[[], NumericStepFunc] = lambda: Euler(), sps: int = 100):
    g = 1e7
    a = 100
    v = np.sqrt((1+2*np.sqrt(2))*(g/a))/2

    sim = Simulation(
        start_balls=[
            make_ball(a * Vector([0, -1]), v * Vector([1, 0]),  start_dv=v**2/a * Vector([0, 1]), step_func=step_func),
            make_ball(a * Vector([0, 1]),  v * Vector([-1, 0]), start_dv=v**2/a * Vector([0, -1]), step_func=step_func),
            make_ball(a * Vector([1, 0]),  v * Vector([0, 1]),  start_dv=v**2/a * Vector([-1, 0]), step_func=step_func),
            make_ball(a * Vector([-1, 0]), v * Vector([0, -1]), start_dv=v**2/a * Vector([1, 0]), step_func=step_func)
        ],
        fields={},
        interactions=[Interaction("gravity", ("Ball", "Ball"),
                                  formula=lambda target, source:
                                  g/(Ball.distance(target[0], source)**2) * target[0].direction(source),
                                  attribute="velocity"
                                  )],
        total_time=5, steps_per_second=sps, field_size=(1000, 600))

    sim_display = SimulationDisplay(sim, fps=50, ball_radius=8)
    sim_display.display_sim()


# test4(step_func=lambda: Method3212(), sps=10)
test0()
