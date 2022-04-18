from __future__ import annotations
from Simulation import *
from math_primitives.ExpSplineFieldRepr import ExpSplineFieldRepr
from math_primitives.SplineFieldRepr import SplineFieldRepr


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
        [Ball(Vector2([15, -100]), Vector2([0, 150])),
         Ball(Vector2([-15, 100]), Vector2([0, -180])),
         Ball(Vector2([-100, 75]), Vector2([162, 0]))],
        {"repellent": Field("repellent", field_repr)},
        [Interaction("Repulsion", ("Ball", "repellent"),
                     lambda target, repellent: 1000 * target[0].position.normalize() * repellent.val(
                         target[0].position),
                     attribute="velocity"),
         Interaction("Diffusion", ("repellent", "repellent"),
                     lambda _, rep: lambda x, y: 50 * (rep.dx2(x, y) + rep.dy2(x, y)))],
        total_time=5, steps_per_second=100, field_size=(1000, 600))

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
        [Ball(Vector([15, -100]), Vector([0, 150])),
         Ball(Vector([-15, 100]), Vector([0, -180])),
         Ball(Vector([-100, 75]), Vector([162, 0]))],
        {"height": Field("height", ExpSplineFieldRepr.from_values((1000, 600), height_values))},
        [Interaction("Gravity", ("Ball", "height"),
                     lambda target, h: -20 * h.gradient(target[0].position.x,
                                                        target[0].position.y),
                     attribute="velocity")],
        total_time=5, steps_per_second=1000, field_size=(1000, 600))

    sim_display = SimulationDisplay(sim, 50, 8, scale=1,
                                    background_field="height",
                                    background_field_res=(60, 60),
                                    background_field_range=(0, 7000))
    sim_display.display_sim()


def test3():
    temp_field = make_field_vals((1000, 600), (50, 50), lambda x, y: 10000 / (x ** 2 + y ** 2 + 1))

    print((temp_field.min(initial=inf), temp_field.sum() / temp_field.size, temp_field.max(initial=-inf)))

    sim = Simulation(
        [Ball(Vector([0, -100]), Vector([0, 150])),
         Ball(Vector([-15, 100]), Vector([10, -180])),
         Ball(Vector([-100, 75]), Vector([162, 10]))],
        {"temperature": Field("temperature", SplineFieldRepr.from_values((1000, 600), temp_field))},
        [Interaction("Too hot!", ("Ball", "temperature"),
                     lambda target, temp: -10000 * temp.gradient(target[0].position.x,
                                                                 target[0].position.y)
                                          - 5000 * temp.dir(target[0].position.x,
                                                            target[0].position.y,
                                                            target[0].velocity.normalize())
                     * target[0].velocity.normalize(),
                     attribute="velocity")],
        total_time=15, steps_per_second=200, field_size=(1000, 600))

    test_point = Vector([10, 10])
    print(sim.fields["temperature"].val(test_point))
    print(sim.fields["temperature"].gradient(test_point.x, test_point.y))

    disp = SimulationDisplay(sim, 50, 8, scale=1,
                             background_field="temperature",
                             background_field_res=(60, 60),
                             background_field_range=(0, 30))
    disp.display_sim()


def make_field_vals(size: Tuple[float, float], res: Tuple[int, int],
                    func: Callable[[float, float], float]):
    cords = np.linspace(-size[0] / 2, size[0] / 2, res[0]), np.linspace(-size[1] / 2, size[1] / 2, res[1])
    values = np.zeros((cords[1].size, cords[0].size))
    for i in range(len(cords[0])):
        for j in range(len(cords[1])):
            values[j, i] = func(cords[0][i], cords[1][j])
    return values


test2()
