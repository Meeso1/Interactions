from __future__ import annotations
from typing import Literal
import pygame as pg
from BFIClasses import *
from math_primitives.SplineFieldRepr import SplineFieldRepr

n: int = 20         # Field resolution (represented as n x n x n array)
width: float = 100  # Field width


class Simulation:

    def __init__(self,
                 start_balls: List[Ball],
                 fields: Dict[str, Field],
                 interactions: List[Interaction],
                 field_res: Tuple[int, int] = (n, n),
                 field_size: Tuple[float, float] = (width, width),
                 border_size: Tuple[float, float] | Literal["auto"] = "auto",
                 border_effect: Literal["repel", "stop", "ignore"] = "repel",
                 steps_per_second: int = 100,
                 start_time: float = 0,
                 total_time: float = 100):

        self.field_res: Tuple[int, int] = field_res
        self.field_size: Tuple[float, float] = field_size

        self.border_size: Tuple[float, float]
        if border_size == "auto":
            self.border_size = field_size
        else:
            self.border_size = border_size

        self.border_effect: Literal["repel", "stop", "ignore"] = border_effect

        self.balls: List[Ball] = start_balls
        self.fields: Dict[str, Field] = fields
        self.interactions: List[Interaction] = interactions

        # Interactions with border
        if not self.border_effect == "ignore":
            border_field = Field("border field",
                                 SplineFieldRepr.from_values(self.field_size, np.zeros((20, 20)), const=True))
            self.fields["border field"] = border_field
        if self.border_effect == "repel":
            border_interaction = Interaction("border repulsion", ("Ball", "border field"),
                                             lambda target, _: self._check_border_repel(target[0]),     # type: ignore
                                             attribute="velocity")
            self.interactions.append(border_interaction)
        elif self.border_effect == "stop":
            border_interaction = Interaction("border friction", ("Ball", "border field"),
                                             lambda target, _: self._check_border_stop(target[0]),      # type: ignore
                                             attribute="velocity")
            self.interactions.append(border_interaction)

        self.dt: float = 1/steps_per_second
        self.total_time: float = total_time
        self._timesteps: NDArray[np.float64] = \
            np.linspace(start_time, self.total_time, int(steps_per_second*self.total_time))

        self.time: float = start_time
        self.step_num: int = 0

    def add_balls(self, new_balls: List[Ball]) -> None:
        if len(new_balls) <= 0:
            return
        for ball in new_balls:
            self.balls.append(ball)

    def _check_border_repel(self, ball: Ball) -> Vector2:
        k = 5
        damp = 1.01

        res = Vector([0, 0])
        if self.border_size is None:
            return res

        pos = ball.position.current
        size = Vector(self.border_size)
        if pos.x < -size.x/2:
            res.x += min(k * (pos.x + size.x / 2) ** 2, abs(ball.velocity.current.x) / (self.dt * damp))
        elif pos.x > size.x/2:
            res.x -= min(k * (pos.x - size.x / 2) ** 2, abs(ball.velocity.current.x) / (self.dt * damp))

        if pos.y < -size.y/2:
            res.y += min(k * (pos.y + size.y / 2) ** 2, abs(ball.velocity.current.y) / (self.dt * damp))
        elif pos.y > size.y/2:
            res.y -= min(k * (pos.y - size.y / 2) ** 2, abs(ball.velocity.current.y) / (self.dt * damp))

        return res

    def _check_border_stop(self, ball: Ball) -> Vector2:
        k = 5
        damp = 2

        res = Vector([0, 0])
        if self.border_size is None:
            return res

        pos = ball.position.current
        vel = ball.velocity.current
        size = Vector(self.border_size)
        if pos.x < -size.x / 2:
            if vel.x < 0:
                res.x += min(k * vel.x ** 2, abs(vel.x)/(self.dt * damp))
        elif pos.x > size.x / 2:
            if vel.x > 0:
                res.x -= min(k * vel.x ** 2, abs(vel.x) / (self.dt * damp))

        if pos.y < -size.y / 2:
            if vel.y < 0:
                res.y += min(k * vel.y ** 2, abs(vel.y)/(self.dt * damp))
        elif pos.y > size.y / 2:
            if vel.y > 0:
                res.y -= min(k * vel.y ** 2, abs(vel.y) / (self.dt * damp))

        return res

    def step(self) -> None:
        if self.step_num >= len(self._timesteps)-1:
            return

        self.step_num += 1

        for interaction in self.interactions:
            interaction.update(self._timesteps[self.step_num], (self.balls, self.fields))

        for ball in self.balls:
            ball.update(self._timesteps[self.step_num])
        for field in self.fields.values():
            field.update(self._timesteps[self.step_num])

        self.time = self._timesteps[self.step_num]

    def simulate(self, max_time: float = inf) -> None:
        if max_time == inf:
            max_time = self.total_time
        while self.time < max_time and self.step_num < len(self._timesteps)-1:
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
                               self.pos_to_screen((ball.position.current.x, ball.position.current.y)),
                               self.ball_radius)
            pg.display.update()

            clock.tick(self.FPS)

    def pos_to_screen(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        return pos[0] * self.scale + self.display_size[0] / 2, pos[1] * self.scale + self.display_size[1] / 2
