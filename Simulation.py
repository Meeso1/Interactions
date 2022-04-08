from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from math import inf
import pygame as pg
from BFIClasses import *

n: int = 20         # Field resolution (represented as n x n x n array)
width: float = 100  # Field width


class Simulation:

    balls: List[Ball]
    fields: Dict[str, Field]
    interactions: List[Interaction]

    field_res: Tuple[int, int]
    field_size: Tuple[float, float]
    border_size: Union[Tuple[float, float], None]
    border_effect: str

    dt: float
    total_time: float
    time: float
    step_num: int

    ball_tree: spatial.KDTree = None

    def __init__(self, start_balls: List[Ball], fields: Dict[str, Field], interactions: List[Interaction],
                 field_res: Tuple[int, int] = (n, n), field_size: Tuple[float, float] = (width, width),
                 border_size: Union[Tuple[float, float], str] = "auto", border_effect: str = "bounce",
                 steps_per_second: int = 100, total_time: float = 100):
        self.field_res = field_res
        self.field_size = field_size
        if border_size == "auto":
            self.border_size = field_size
        else:
            self.border_size = border_size
        if border_effect not in ("bounce", "stop", "contain"):
            raise Exception(f"Unrecognised border effect: {border_effect}", border_effect)
        else:
            self.border_effect = border_effect

        self.balls = start_balls
        self.fields = fields
        self.interactions = interactions

        # Setup attr_derivatives for balls
        for ball in start_balls:
            ball.attr_derivatives["velocity"] = []
            if ball.attributes is None:
                continue
            for attr in ball.attributes.keys():
                ball.attr_derivatives[attr] = []

        # Setup field res and size
        for field in fields.values():
            if field.res is None:
                field.res = field.values.shape
            if field.size is None:
                field.size = self.field_size

        self.dt = 1/steps_per_second
        self.total_time = total_time

        self.time = 0
        self.step_num = 0

        if len(self.balls) > 0:
            self.ball_tree = Ball.make_tree(self.balls)

    def add_balls(self, new_balls: List[Ball]) -> None:
        if len(new_balls) <= 0:
            return
        for ball in new_balls:
            self.balls.append(ball)
            ball.attr_derivatives["velocity"] = []
            if ball.attributes is None:
                continue
            for attr in ball.attributes.keys():
                ball.attr_derivatives[attr] = []
        self.ball_tree = Ball.make_tree(self.balls)

    def add_fields(self, fields: List[Field]) -> None:
        for field in fields:
            self.fields[field.name] = field
            if field.res is None:
                field.res = field.values.shape
            if field.size is None:
                field.size = self.field_size

    def add_interactions(self, interactions: List[Interaction]) -> None:
        for interaction in interactions:
            self.interactions.append(interaction)

    def check_border(self, ball: Ball) -> None:
        if self.border_size is None:
            return

        if ball.position[0] > self.border_size[0]/2:
            ball.position[0] = self.border_size[0] / 2
            if self.border_effect == "bounce":
                ball.velocity[0] = -ball.velocity[0]
            elif self.border_effect == "stop":
                ball.velocity[0] = 0

        elif ball.position[0] < -self.border_size[0]/2:
            ball.position[0] = -self.border_size[0] / 2
            if self.border_effect == "bounce":
                ball.velocity[0] = -ball.velocity[0]
            elif self.border_effect == "stop":
                ball.velocity[0] = 0

        if ball.position[1] > self.border_size[1]/2:
            ball.position[1] = self.border_size[1] / 2
            if self.border_effect == "bounce":
                ball.velocity[1] = -ball.velocity[1]
            elif self.border_effect == "stop":
                ball.velocity[1] = 0

        elif ball.position[1] < -self.border_size[1]/2:
            ball.position[1] = -self.border_size[1] / 2
            if self.border_effect == "bounce":
                ball.velocity[1] = -ball.velocity[1]
            elif self.border_effect == "stop":
                ball.velocity[1] = 0

    def step(self) -> None:
        if self.time >= self.total_time:
            return

        Interaction.time = self.time
        for ball in self.balls:
            ball.update(self.dt)
            self.check_border(ball)
        if len(self.balls) > 0:
            self.ball_tree = Ball.make_tree(self.balls)
        for field in self.fields.values():
            field.update(self.dt)
        for interaction in self.interactions:
            interaction.update(self.dt, (self.balls, self.fields), self.ball_tree)

        # Update attrs and vals based on derivatives
        for ball in self.balls:
            ball.update_attrs(self.dt)
        for field in self.fields.values():
            field.update_val(self.dt)

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

    scale: float

    simulation: Simulation
    FPS: int
    frame_time: float
    ball_radius: float

    time: float
    frame_num: int

    background_field: str
    background_field_res: Tuple[int, int]

    def __init__(self, simulation: Simulation, fps: int = -1,
                 ball_radius: float = 10,
                 display_size: Tuple[int, int] = (1000, 600), scale: float = 1,
                 background_field: str = None,
                 background_field_res: Tuple[int, int] = (60, 60)):
        if fps > 1 / simulation.dt:
            fps = 1 / simulation.dt
        self.FPS = fps
        self.frame_time = 1 / self.FPS
        self.simulation = simulation

        self.time = 0
        self.frame_num = 0
        self.ball_radius = ball_radius

        self.display_size = display_size
        self.scale = scale
        self.background_field = background_field
        self.background_field_res = background_field_res

    def display_sim(self) -> None:
        pg.init()
        display = pg.display.set_mode(self.display_size)
        clock = pg.time.Clock()

        while True:
            # Handle pygame events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return

            # Advance the simulation
            if self.simulation.time < self.simulation.total_time:
                self.simulation.simulate(self.time + self.frame_time)
                self.time += self.frame_time
                self.frame_num += 1

            display.fill(self.background_color)

            # draw background field
            if self.background_field is not None:
                field = self.simulation.fields[self.background_field]
                avg = field.values.sum()/field.values.size
                field_disp_size = int(field.size[0] * self.scale), int(field.size[1] * self.scale)
                values = field.vals(np.linspace(-field.size[0]/2, field.size[0]/2, self.background_field_res[1]),
                                    np.linspace(-field.size[1]/2, field.size[1]/2, self.background_field_res[0]))
                shades = 1 - 1 / (values / avg + 1)
                p_color = (shades*255).astype(np.ubyte)
                colors = np.dstack([(255*np.sin(shades*np.pi/2)).astype(np.ubyte),
                                    (255*(np.sin(shades*np.pi/2)) + np.cos(shades*np.pi/2)/2).astype(np.ubyte),
                                    (255*np.cos(shades*np.pi/2)).astype(np.ubyte)])
                rect_dim = field_disp_size[0]/self.background_field_res[0], \
                    field_disp_size[1]/self.background_field_res[1]
                center = self.pos_to_screen((0, 0))
                for i in range(self.background_field_res[0]):
                    for j in range(self.background_field_res[1]):
                        pg.draw.rect(display, colors[i][j],
                                     pg.Rect((rect_dim[0]*i + center[0] - field_disp_size[0]/2,
                                             rect_dim[1]*j + center[1] - field_disp_size[1]/2),
                                             np.ceil(rect_dim)))

            # Center dot
            pg.draw.circle(display, (0, 0, 0), self.pos_to_screen((0, 0)), 3)

            # Display balls
            for ball in self.simulation.balls:
                pg.draw.circle(display, self.ball_color,
                               self.pos_to_screen((ball.position[0], ball.position[1])),
                               self.ball_radius)
            pg.display.update()

            clock.tick(self.FPS)

    def pos_to_screen(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        return pos[0] * self.scale + self.display_size[0] / 2, pos[1] * self.scale + self.display_size[1] / 2
