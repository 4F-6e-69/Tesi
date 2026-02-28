from abc import ABC, abstractmethod
from src.obj.shape.shape import Shape

from matplotlib import pyplot as plt
from numpy import typing as nptyping
from src.utils import ArrayLike

class ParametricCurve(Shape, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def point_at(self, t: ArrayLike) -> nptyping.NDArray[np.float64]:
        pass
    @property
    @abstractmethod
    def t_range(self) -> tuple[float, float]:
        pass

    def draw(self, ax: plt.Axes = None, points: ArrayLike = None, show: bool = False, **kwargs) -> plt.Axes | None:
        if self.closure is None:
            self.discretize()

        return super().draw(ax, points, show, **kwargs)