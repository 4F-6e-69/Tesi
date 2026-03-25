from dataclasses import dataclass
from typing import Union, Optional
from typing import List, Literal

import numpy as np
from numpy import typing as npt

CoordsType = Union[int, float, np.number]
ArrayLike = Union[npt.NDArray[np.number], npt.ArrayLike, np.ndarray]

from pygarp.core.models.shapes.circle import Circle
from pygarp.core.models.shapes.rectangle import Rectangle
from pygarp.core.models.shapes.regular_polygon import RegularPolygon
from pygarp.core.models.shapes.closed_spline import ClosedSpline
from pygarp.core.models.shapes.shape.core import Shape

__Cache = Literal["all", "bounds", "area", "length", "barycenter", "step", "closure"]
Caches = Union[List[__Cache], __Cache]
TransformationRef = Literal["origin", "center"]
DiscretizationMethod = Optional[Literal["adaptive", "uniform", "none"]]
ShapeType = Literal["shape", "spline", "circle", "rectangle", "regular_polygon"]
Shapes = Union[Shape, ClosedSpline, Circle, Rectangle, RegularPolygon]

FillType = Literal["grid", "linear"]


@dataclass(frozen=True)
class EpsConfig:
    __eps04: float = 1e-04
    __eps05: float = 1e-05
    __eps06: float = 1e-06
    __eps08: float = 1e-08
    __eps10: float = 1e-10
    __eps11: float = 1e-11
    __eps12: float = 1e-12
    __eps13: float = 1e-13
    __eps14: float = 1e-14
    __eps15: float = 1e-15
    __eps16: float = np.finfo(np.float64).eps

    @property
    def eps04(self) -> float:
        return self.__eps04

    @property
    def eps05(self) -> float:
        return self.__eps05

    @property
    def eps06(self) -> float:
        return self.__eps06

    @property
    def eps08(self) -> float:
        return self.__eps08

    @property
    def eps10(self) -> float:
        return self.__eps10

    @property
    def eps11(self) -> float:
        return self.__eps11

    @property
    def eps12(self) -> float:
        return self.__eps12

    @property
    def eps13(self) -> float:
        return self.__eps13

    @property
    def eps14(self) -> float:
        return self.__eps14

    @property
    def eps15(self) -> float:
        return self.__eps15

    @property
    def eps16(self) -> float:
        return self.__eps16


Eps = EpsConfig()
