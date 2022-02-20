from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from scipy import spatial
from math import inf
import pygame as pg

n: int = 20         # Field resolution (represented as n x n x n array)
width: float = 100  # Field width


class Ball:

    # static:
    # balls: List[Ball] = []
    # tree: spatial.KDTree = None

    # instance:
    position: np.ndarray
    velocity: np.ndarray
    attributes: Dict[str, Any]
    trace: List[Tuple[np.ndarray, np.ndarray]]  # position and velocity log for plotting

    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = position
        self.velocity = velocity
        self.attributes = {}
        self.trace = []

        # Ball.balls.append(self)

    def update(self, dt: float) -> None:
        self.position += self.velocity * dt
        self.trace.append((self.position, self.velocity))

    # @staticmethod
    # def update_all(dt: float) -> None:
    #     for ball in Ball.balls:
    #        ball.update(dt)

    @staticmethod
    def make_tree(balls: List[Ball]) -> spatial.KDTree:
        pos = np.array([b.position for b in balls])
        return spatial.KDTree(pos)

    @staticmethod
    def distance(a: Ball, b: Ball) -> float:
        return np.linalg.norm(a.position - b.position)


class Field:

    # static:
    # fields: Dict[str, Field] = {}

    # instance:
    name: str
    values: np.ndarray
    trace: List[np.ndarray]     # Field values log for plotting and analysis

    def __init__(self, name: str, values: np.ndarray) -> None:
        self.name = name
        self.values = values

        # Field.fields[self.name] = self

    def update(self, dt: float):
        self.trace.append(self.values)

    # @staticmethod
    # def update_all(dt: float):
    #    for (key, value) in Field.fields:
    #        value.update(dt)


class Interaction:

    Target = Union[Tuple[Ball, str], Field]
    Source = Union[Ball, Field]
    Derivative_Func = Callable[[Target, Source], Union[float, np.ndarray]]

    # static:
    interactions: List[Interaction] = []

    # instance:
    name: str
    targets: Tuple[str, str]
    formula: Derivative_Func
    attribute: str
    radius: float

    def __init__(self,
                 name: str,
                 between: Tuple[str, str],
                 formula: Derivative_Func,  # f: a,b => da/dt
                 radius: Optional[float] = inf,
                 attribute: Optional[str] = None) -> None:
        self.name = name
        self.targets = between
        self.radius = radius
        self.attribute = attribute
        self.formula = formula  # derivative of sth.

        Interaction.interactions.append(self)

    def update(self, dt: float, targets: Tuple[List[Ball], Dict[str, Field]], ball_tree: spatial.KDTree = None):
        (balls, fields) = targets
        if self.targets == ("Ball", "Ball"):
            if ball_tree is not None:
                res = ball_tree.query_ball_tree(ball_tree, self.radius)
            else:
                res = [[index for index in range(len(balls))
                        if self.radius == inf or Ball.distance(ball, balls[index]) < self.radius]
                       for ball in balls]
            for i in range(len(res)):
                for j in res[i]:
                    if j != i:
                        self.update_val((balls[i], self.attribute),
                                        balls[j],
                                        self.formula, dt)

        elif self.targets[0] == "Ball":
            for ball in balls:
                self.update_val((ball, self.attribute),
                                fields[self.targets[1]],
                                self.formula, dt)

        elif self.targets[1] == "Ball":
            for ball in balls:
                self.update_val(fields[self.targets[0]],
                                ball, self.formula, dt)

        else:
            self.update_val(fields[self.targets[0]],
                            fields[self.targets[1]],
                            self.formula, dt)

    # @staticmethod
    # def update_all(dt: float):
    #     for interaction in Interaction.interactions:
    #         interaction.update(dt)

    @classmethod
    def update_val(cls, target: Target, source: Source, f: Derivative_Func, dt: float):
        if isinstance(target, Field):
            target.values += f(target, source) * dt   # TODO: Improve

        else:   # target is Ball
            if target[1] == "velocity":
                target[0].velocity += f(target, source) * dt
            else:
                target[0].attributes[target[1]] += f(target, source) * dt


class Simulation:

    balls: List[Ball]
    fields: Dict[str, Field]
    interactions: List[Interaction]

    field_res: int
    field_width: float

    dt: float
    total_time: float
    time: float
    step_num: int

    ball_tree: spatial.KDTree

    def __init__(self, start_balls: List[Ball], fields: Dict[str, Field], interactions: List[Interaction],
                 field_res: int = n, field_points_width: float = width,
                 steps_per_second: int = 100, total_time: float = 100):
        self.balls = start_balls
        self.fields = fields
        self.interactions = interactions

        self.field_res = field_res
        self.field_width = field_points_width

        self.dt = 1/steps_per_second
        self.total_time = total_time

        self.time = 0
        self.step_num = 0

        self.ball_tree = Ball.make_tree(self.balls)

    def add_balls(self, new_balls: List[Ball]):
        for ball in new_balls:
            self.balls.append(ball)
        self.ball_tree = Ball.make_tree(self.balls)

    def add_fields(self, fields: List[Field]):
        for field in fields:
            self.fields[field.name] = field

    def add_interactions(self, interactions: List[Interaction]):
        for interaction in interactions:
            self.interactions.append(interaction)

    def step(self) -> None:
        if self.time >= self.total_time:
            return

        for ball in self.balls:
            ball.update(self.dt)
        self.ball_tree = Ball.make_tree(self.balls)
        for field in self.fields.values():
            field.update(self.dt)
        for interaction in self.interactions:
            interaction.update(self.dt, (self.balls, self.fields), self.ball_tree)

        self.time += self.dt
        self.step_num += 1

    def simulate(self, max_time: float = inf) -> None:
        if max_time == inf:
            max_time = self.total_time
        while self.time < max_time:
            self.step()


class SimulationDisplay:

    Color = Tuple[int, int, int]
    background_color: Color = (255, 255, 255)
    ball_color: Color = (255, 0, 255)
    display_size: Tuple[int, int] = (1000, 600)

    simulation: Simulation
    FPS: int
    frame_time: float
    ball_radius: float

    time: float
    frame_num: int

    def __init__(self, simulation: Simulation, fps: int = -1, ball_radius: float = 10):
        if fps > 1 / simulation.dt:
            fps = 1 / simulation.dt
        self.FPS = fps
        self.frame_time = 1 / self.FPS
        self.simulation = simulation

        self.time = 0
        self.frame_num = 0
        self.ball_radius = ball_radius

    def display_sim(self):
        pg.init()
        display = pg.display.set_mode(self.display_size)
        clock = pg.time.Clock()

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return

            if self.simulation.time < self.simulation.total_time:
                self.simulation.simulate(self.time + self.frame_time)
                self.time += self.frame_time
                self.frame_num += 1

            display.fill(self.background_color)
            for ball in self.simulation.balls:
                pg.draw.circle(display, self.ball_color, (ball.position[0], ball.position[1]), self.ball_radius)
            pg.display.update()

            clock.tick(self.FPS)
