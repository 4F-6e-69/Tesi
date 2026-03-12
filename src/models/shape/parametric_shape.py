from abc import ABC, abstractmethod

import numpy as np
from numpy import typing as nptyping

from src.models.shape.generic_shape import Shape
from src.utils import ArrayLike

class ParametricShape(ABC, Shape):
    @abstractmethod
    def point_at(self, t: ArrayLike) -> nptyping.NDArray[np.float64]:
        pass
    @property
    @abstractmethod
    def t_range(self) -> tuple[float, float]:
        pass

    def _calc_max_discretization_step(self, **kwargs) -> float:
        return self.length / 3