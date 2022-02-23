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

    dt: float
    total_time: float
    time: float
    step_num: int

    ball_tree: spatial.KDTree

    def __init__(self, start_balls: List[Ball], fields: Dict[str, Field], interactions: List[Interaction],
                 field_res: Tuple[int, int] = (n, n), field_size: Tuple[float, float] = (width, width),
                 steps_per_second: int = 100, total_time: float = 100):
        self.field_res = field_res
        self.field_size = field_size

        self.balls = start_balls
        self.fields = fields
        self.interactions = interactions

        for field in fields.values():
            if field.res is None:
                field.res = field.values.shape
            if field.size is None:
                field.size = self.field_size

        self.dt = 1/steps_per_second
        self.total_time = total_time

        self.time = 0
        self.step_num = 0

        self.ball_tree = Ball.make_tree(self.balls)

    def add_balls(self, new_balls: List[Ball]) -> None:
        for ball in new_balls:
            self.balls.append(ball)
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

            # Display background field
            # if self.background_field is not None:
            #     field = self.simulation.fields[self.background_field]
            #     field_disp_size = int(field.size[0] * self.scale), int(field.size[1] * self.scale)
            #     values = field.vals(np.linspace(-field.size[0]/2, field.size[0]/2, field_disp_size[1]),
            #                         np.linspace(-field.size[1]/2, field.size[1]/2, field_disp_size[0]))
            #    shades = 1/(values*10 + 1)
            #    p_color = (shades*255).astype(np.ubyte)
            #    colors = np.dstack([p_color, np.ones(p_color.shape)*255, p_color])
            #    surf = pg.surfarray.make_surface(colors)
            #    center = self.pos_to_screen((0, 0))
            #    display.blit(surf, (center[0] - field_disp_size[0]/2, center[1] - field_disp_size[1]/2))

            if self.background_field is not None:
                field = self.simulation.fields[self.background_field]
                field_disp_size = int(field.size[0] * self.scale), int(field.size[1] * self.scale)
                values = field.vals(np.linspace(-field.size[0]/2, field.size[0]/2, self.background_field_res[1]),
                                    np.linspace(-field.size[1]/2, field.size[1]/2, self.background_field_res[0]))
                shades = 1 / (values * 10 + 1)
                p_color = (shades*255).astype(np.ubyte)
                colors = np.dstack([p_color, np.ones(p_color.shape)*255, p_color])
                rect_dim = field_disp_size[0]/self.background_field_res[0], \
                    field_disp_size[1]/self.background_field_res[1]
                center = self.pos_to_screen((0, 0))
                for i in range(self.background_field_res[0]):
                    for j in range(self.background_field_res[1]):
                        pg.draw.rect(display, colors[i][j],
                                     pg.Rect((rect_dim[0]*i + center[0] - field_disp_size[0]/2,
                                             rect_dim[1]*j + center[1] - field_disp_size[1]/2),
                                             np.ceil(rect_dim)))

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
