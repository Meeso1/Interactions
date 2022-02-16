from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import random
import pygame as pg
from BFIClasses import *


def simulate(time: float, steps: int):
    dt: float = time / steps
    for i in range(steps):
        Ball.update_all(dt)
        Field.update_all(dt)
        Interaction.update_all(dt)


def sim_display(time: float, framerate: int):
    pg.init()
    display = pg.display.set_mode((1000, 600))
    clock = pg.time.Clock()
    dt: float = 1 / framerate

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return

        Ball.update_all(dt)
        Field.update_all(dt)
        Interaction.update_all(dt)

        display.fill((255, 255, 255))
        for ball in Ball.balls:
            pg.draw.circle(display, (255, 0, 255), (ball.position[0], ball.position[1]), 10)
        pg.display.update()

        clock.tick(framerate)


Ball(np.array([200, 300], dtype=float), np.array([100, 20], dtype=float))
sim_display(10, 50)
