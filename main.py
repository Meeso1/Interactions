from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import random
from Simulation import *


def test1():
    repellent_values = np.ones((20, 20), dtype=float)
    x_lim = repellent_values.shape[0]
    y_lim = repellent_values.shape[1]
    field_center = (1, 1.5)
    for i in range(x_lim):
        for j in range(y_lim):
            repellent_values[i, j] = 10 * 1 / (25 * np.linalg.norm((i - x_lim / 2 + 0.5 - field_center[0],
                                                                    j - y_lim / 2 + 0.5 - field_center[1])) ** 2 + 1)

    print((repellent_values.min(), repellent_values.sum()/repellent_values.size, repellent_values.max()))

    sim = Simulation(
        [Ball(np.array([15, -100], dtype=float), np.array([0, 150], dtype=float)),
         Ball(np.array([-15, 100], dtype=float), np.array([0, -180], dtype=float)),
         Ball(np.array([-100, 75], dtype=float), np.array([162, 0], dtype=float))],
        {"repellent": Field("repellent", repellent_values, (1000, 600))},
        [Interaction("Repulsion", ("Ball", "repellent"),
                     lambda target, repellent: 1000 * normalized(target[0].position) * repellent.val(
                         target[0].position),
                     attribute="velocity"),
         Interaction("Diffusion", ("repellent", "repellent"),
                     lambda _, rep: lambda x, y: 50 * (rep.dx2_mat(x, y) + rep.dy2_mat(x, y)))],
        total_time=15, steps_per_second=100, field_size=(1000, 600))

    disp = SimulationDisplay(sim, 50, 8, scale=1, background_field="repellent", background_field_res=(50, 50))
    disp.display_sim()


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
                val += height / (10 * np.linalg.norm((i - x_lim / 2 + 0.5 - peak[0],
                                                       j - y_lim / 2 + 0.5 - peak[1])) ** 2 + 1)
            height_values[i, j] = val

    print((height_values.min(), height_values.sum()/height_values.size, height_values.max()))

    sim = Simulation(
        [Ball(np.array([15, -100], dtype=float), np.array([0, 150], dtype=float)),
         Ball(np.array([-15, 100], dtype=float), np.array([0, -180], dtype=float)),
         Ball(np.array([-100, 75], dtype=float), np.array([162, 0], dtype=float))],
        {"height": Field("height", height_values, (1000, 600))},
        [Interaction("Gravity", ("Ball", "height"),
                     lambda target, height: -10 * height.dv(target[0].position),
                     attribute="velocity")],
        total_time=15, steps_per_second=1000, field_size=(1000, 600))

    disp = SimulationDisplay(sim, 50, 8, scale=1, background_field="height", background_field_res=(60, 60))
    disp.display_sim()


def test3():
    temp_field = make_field_vals((1000, 600), (50, 50), lambda x, y: 10000/(x**2+y**2+1))

    print((temp_field.min(), temp_field.sum()/temp_field.size, temp_field.max()))

    sim = Simulation(
        [Ball(np.array([0, -100], dtype=float), np.array([0, 150], dtype=float)),
         Ball(np.array([-15, 100], dtype=float), np.array([10, -180], dtype=float)),
         Ball(np.array([-100, 75], dtype=float), np.array([162, 10], dtype=float))],
        {"temperature": Field("temperature", temp_field, (1000, 600))},
        [Interaction("Too hot!", ("Ball", "temperature"),
                     lambda target, temp: -10000 * temp.gradient(target[0].position),
                     attribute="velocity")],
        total_time=15, steps_per_second=1000, field_size=(1000, 600))

    test_point = np.array([10, 10])
    print(sim.fields["temperature"].val(test_point))
    print(sim.fields["temperature"].gradient(test_point))

    disp = SimulationDisplay(sim, 50, 8, scale=1, background_field="temperature", background_field_res=(60, 60))
    disp.display_sim()


def make_field_vals(size: Tuple[float, float], res: Tuple[int, int],
                    func: Callable[[float, float], float]):
    cords = np.linspace(-size[0] / 2, size[0] / 2, res[0]), np.linspace(-size[1] / 2, size[1] / 2, res[1])
    values = np.zeros((cords[1].size, cords[0].size))
    for i in range(len(cords[0])):
        for j in range(len(cords[1])):
            values[j, i] = func(cords[0][i], cords[1][j])
    return values


test3()
