import numpy as np
import typing
import abc


class Ball:

    balls = []

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.attributes = {}

        Ball.balls.append(self)

    def update(self, dt: float):
        self.position += self.velocity * dt
