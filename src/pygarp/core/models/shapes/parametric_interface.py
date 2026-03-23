from abc import ABC, abstractmethod

import numpy as np
from numpy import typing as npt

from pygarp.core.models.shapes.shape.core import Shape
from pygarp.core.models.commons import ArrayLike

class ParametricShape(Shape, ABC):
    @abstractmethod
    def point_at(self, t: ArrayLike) -> npt.NDArray[np.float64]:
        pass
    @property
    @abstractmethod
    def t_range(self) -> tuple[float, float]:
        pass

    def _calc_step_max(self) -> float:
        return self.length / 3