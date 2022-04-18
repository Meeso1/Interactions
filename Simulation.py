from __future__ import annotations
from typing import Literal
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
    border_size: Tuple[float, float] | None
    border_effect: Literal["bounce", "stop", "contain"]

    dt: float
    total_time: float
    time: float
    step_num: int

    def __init__(self,
                 start_balls: List[Ball],
                 fields: Dict[str, Field],
                 interactions: List[Interaction],
                 field_res: Tuple[int, int] = (n, n),
                 field_size: Tuple[float, float] = (width, width),
                 border_size: Tuple[float, float] | Literal["auto"] = "auto",
                 border_effect: Literal["bounce", "stop", "contain"] = "bounce",
                 steps_per_second: int = 100,
                 total_time: float = 100):

        self.field_res: Tuple[int, int] = field_res
        self.field_size: Tuple[float, float] = field_size

        self.border_size: Tuple[float, float]
        if border_size == "auto":
            self.border_size = field_size
        else:
            self.border_size = border_size

        self.border_effect: Literal["bounce", "stop", "contain"] = border_effect

        self.balls: List[Ball] = start_balls
        self.fields: Dict[str, Field] = fields
        self.interactions: List[Interaction] = interactions

        # Setup attr_derivatives for balls
        for ball in start_balls:
            ball.attr_derivatives["velocity"] = []
            if ball.attributes is None:
                continue
            for attr in ball.attributes.keys():
                ball.attr_derivatives[attr] = []

        self.dt: float = 1/steps_per_second
        self.total_time: float = total_time

        self.time: float = 0
        self.step_num: int = 0

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

        for ball in self.balls:
            ball.init_attr_derivatives(self.time)

        for interaction in self.interactions:
            interaction.update(self.dt, (self.balls, self.fields))

        for ball in self.balls:
            ball.update(self.dt)
            self.check_border(ball)
        for field in self.fields.values():
            field.update(self.dt)

        self.time += self.dt
        self.step_num += 1

    def simulate(self, max_time: float = inf) -> None:
        if max_time == inf:
            max_time = self.total_time
        while self.time < max_time:
            self.step()


Color: TypeAlias = Tuple[int, int, int]


class SimulationDisplay:

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

    background_field: str | None
    background_field_res: Tuple[int, int]
    background_field_range: Tuple[float, float]

    def __init__(self,
                 simulation: Simulation,
                 fps: int = -1,
                 ball_radius: float = 10,
                 display_size: Tuple[int, int] = (1000, 600),
                 scale: float = 1,
                 background_field: str | None = None,
                 background_field_res: Tuple[int, int] = (60, 60),
                 background_field_range: Tuple[float, float] = (0, 100)):

        if fps > 1 / simulation.dt or fps == -1:
            fps = np.floor(1 / simulation.dt)

        self.FPS: int = fps
        self.frame_time: float = 1 / self.FPS
        self.simulation: Simulation = simulation

        self.time: float = 0
        self.frame_num: int = 0
        self.ball_radius: float = ball_radius

        self.display_size: Tuple[int, int] = display_size
        self.scale: float = scale
        self.background_field: str | None = background_field
        self.background_field_res: Tuple[int, int] = background_field_res
        self.background_field_range: Tuple[float, float] = background_field_range

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
                field_real_size = field.values.get_size()
                min_v = self.background_field_range[0]     # vals.max()
                max_v = self.background_field_range[1]     # vals.min()

                # avg = vals.sum()/vals.size
                avg_range = (max_v + min_v)/2
                field_display_size = int(field_real_size[0] * self.scale), int(field_real_size[1] * self.scale)
                values = field.vals(
                    np.linspace(-field_real_size[0]/2, field_real_size[0]/2, self.background_field_res[1]),
                    np.linspace(-field_real_size[1]/2, field_real_size[1]/2, self.background_field_res[0]))
                shades = np.arctan((values - avg_range)/(max_v - min_v)*5)
                # p_color = (shades*255).astype(np.ubyte)
                colors = np.dstack([(255*(np.sin(shades*np.pi/2)+1)/2).astype(np.ubyte),
                                    (255*(np.sin(shades*np.pi/2) +
                                          np.cos(shades*np.pi/2) +
                                          1.42)/(1.42*2)).astype(np.ubyte),
                                    (255*(np.cos(shades*np.pi/2)+1)/2).astype(np.ubyte)])
                rect_dim = field_display_size[0]/self.background_field_res[0], \
                    field_display_size[1]/self.background_field_res[1]
                center = self.pos_to_screen((0, 0))
                for i in range(self.background_field_res[0]):
                    for j in range(self.background_field_res[1]):
                        pg.draw.rect(display, colors[i][j],
                                     pg.Rect((rect_dim[0]*i + center[0] - field_display_size[0]/2,
                                             rect_dim[1]*j + center[1] - field_display_size[1]/2),
                                             list(np.ceil(rect_dim))))

            # Center dot
            pg.draw.circle(display, (0, 0, 0), self.pos_to_screen((0, 0)), 3)

            # Display balls
            for ball in self.simulation.balls:
                pg.draw.circle(display, self.ball_color,
                               self.pos_to_screen((ball.position.x, ball.position.y)),
                               self.ball_radius)
            pg.display.update()

            clock.tick(self.FPS)

    def pos_to_screen(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        return pos[0] * self.scale + self.display_size[0] / 2, pos[1] * self.scale + self.display_size[1] / 2
