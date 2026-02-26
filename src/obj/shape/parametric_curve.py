from abc import ABC, abstractmethod

import numpy as np

from src.obj.plane.working_plane import WorkingSpace
from src.obj.shape.shape import Shape


class ParametricCurve(Shape, ABC):
    def __init__(self, origin: np.typing.ArrayLike = (0, 0), step: float = 0.1):
        super().__init__()

        self.origin = origin
        self.closure_step = step

    @abstractmethod
    def point_at(self, t: np.typing.ArrayLike) -> np.typing.NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def t_range(self) -> tuple[float, float]:
        pass
    @property
    def is_closed_loop(self) -> bool:
        return False
    def _discretization(self):
        t_start, t_end = self.t_range
        total_range = abs(t_end - t_start)

        if self.closure_step < self.tolerance:
            raise ValueError("Lo step deve essere positivo")

        num_points = int(np.ceil(total_range / self.closure_step))
        if num_points < 2: num_points = 2

        use_endpoint = not self.is_closed_loop
        t_values = np.linspace(t_start, t_end, num_points, endpoint=use_endpoint)
        generated_points = self.point_at(t_values)

        self._closure = generated_points

    def draw(self):
        self._discretization()

        super().draw()