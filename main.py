from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import random
from Simulation import *


repellent_values = np.ones((20, 20), dtype=float)
x_lim = repellent_values.shape[0]
y_lim = repellent_values.shape[1]
for i in range(x_lim):
    for j in range(y_lim):
        repellent_values[i, j] = 1/(np.linalg.norm((i - x_lim/2 + 0.5, j - y_lim/2 + 0.5))**2 + 1)

sim = Simulation(
    [Ball(np.array([5, -100], dtype=float), np.array([0, 50], dtype=float)),
     Ball(np.array([-5, 100], dtype=float), np.array([0, -50], dtype=float))],
    {"repellent": Field("repellent", repellent_values, (300, 100))},
    [Interaction("Gravity", ("Ball", "Ball"),
                 lambda target, other_ball: (50000/Ball.distance(target[0], other_ball)**2)
                 * target[0].direction(other_ball),
                 attribute="velocity"),
     Interaction("Repulsion", ("Ball", "repellent"),
                 lambda target, repellent: 1000 * normalized(target[0].position) * repellent.val(target[0].position),
                 attribute="velocity")],
    total_time=15, steps_per_second=100)

disp = SimulationDisplay(sim, 50, 8, scale=3, background_field="repellent", background_field_res=(70, 70))
disp.display_sim()
