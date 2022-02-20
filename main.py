from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import random
from BFIClasses import *

sim = Simulation(
    [Ball(np.array([445, 300], dtype=float), np.array([0, 0], dtype=float)),
     Ball(np.array([555, 300], dtype=float), np.array([0, 0], dtype=float)),
     Ball(np.array([500, 440], dtype=float), np.array([0, -10], dtype=float))],
    {},
    [Interaction("Gravity", ("Ball", "Ball"),
                 lambda target, other_ball: (-5000000/Ball.distance(target[0], other_ball)**3)
                                            * (target[0].position - other_ball.position),
                 attribute="velocity", radius=100)],
    total_time=15, steps_per_second=1000)

disp = SimulationDisplay(sim, 50, 8)
disp.display_sim()
