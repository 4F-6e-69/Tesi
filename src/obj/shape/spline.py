from typing import Optional, Tuple

import matplotlib.pyplot as plt

from src.utils import *
import numpy as np
from numpy import typing as nptyping, dtype
from scipy.interpolate import splprep, splev, interp1d
from matplotlib import pyplot as plt
import scipy
from src.obj.shape.parametric_curve import ParametricCurve
from src.obj.shape.shape import Shape

class ClosedSpline(ParametricCurve):
    def __init__(self, points, smoothness: float = 0.25, high_definition_points: int = 10000):
        new_points_array = validate_array_of_2d_coordinates(points)
        points_x = new_points_array[:, 0]
        points_y = new_points_array[:, 1]

        spline = splprep(np.vstack((points_x, points_y), dtype=np.float64), s=smoothness)
        self.__tck = spline[0]
        self.__u = spline[1]

        self._high_definition_u = np.linspace(0, 1, high_definition_points)
        self._high_definition_closure = splev(points_x, self.high_definition_u)

        super().__init__()

    def _calc_min_closure_step(self):
        self._min_closure_step = float((1 / len(self._high_definition_closure)) * self.length)
    def _calc_max_closure_step(self):
        self._max_closure_step = float(self.length / 3)

    @property
    def high_definition_u(self):
        return self._high_definition_u
    @property
    def t_range(self) -> tuple[float, float]:
        return 0.0, 1.0
    @property
    def high_definition_closure(self):
        return self._high_definition_closure

    @property
    def length(self) -> float:
        if self._length is None:
            self._ensure_valid_closure("lunghezza")
            self._length = Shape.calc_length(self.high_definition_closure)

        return self._length
    @property
    def area_hd(self) -> float:
        if self._area is None:
            self._ensure_valid_closure("area")
            self._area = Shape.calc_area(self.high_definition_closure)

        return self._area
    @property
    def bounding_box_hd(self) -> Tuple[float, float, float, float, float, float]:
        if self._bounding_box is None:
            self._ensure_valid_closure("bounding box")
            self._bounding_box = Shape.calc_bounding_box(self.high_definition_closure)

        return self._bounding_box
    @property
    def barycenter_hd(self) -> nptyping.NDArray[np.float64]:
        if self._barycenter is None:
            self._ensure_valid_closure("baricentro")
            self._barycenter = Shape.calc_barycenter(self.high_definition_closure)

        return self._barycenter

    def point_at(self, t_values: nptyping.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        x, y = splev(t_values, self.__tck)
        return np.column_stack((x, y))

    def _discretization(self, ):
        t_start, t_end = self.t_range
        total_range = float(abs(t_end - t_start))

        self._ensure_valid_closure("contorno")

        num_points = np.ceil(self.length / self.closure_step)

        diffs = self.high_definition_closure - np.roll(self.high_definition_closure, -1)
        dists = np.linalg.norm(diffs, axis=1)
        cumulative_distance = np.insert(np.cumsum(dists), 0, 0)
        total_dist = cumulative_distance[-1]
        u_from_dist = interp1d(cumulative_distance, self.high_definition_u)

        t_target =  np.linspace(0, total_dist, num_points)
        t_values = u_from_dist(t_target)
        generated_points = self.point_at(t_values)

        self._closure = generated_points



    def draw(self, ax: plt.Axes = None, show: bool = False, **kwargs):
        if self.closure is None or len(self.closure) < 3:
            warnings.warn("Nessun contorno disponibile per il disegno.") # Al posto di print()
            return ax

        if ax is None:
            ax = Shape._get_styled_axis()

        x, y = self.closure[:, 0].copy(), self.closure[:, 1].copy()
        high_definition_x, high_definition_y = self.high_definition_closure[:, 0].copy(), self.high_definition_closure[:, 1].copy()

        ax.plot(high_definition_x, high_definition_y, '--', color='grey')
        ax.plot(x, y, **kwargs)

        if show:
            plt.show()

        return ax
'''
class Spline(ParametricCurve):
    def __init__(self, points: np.typing.ArrayLike, divider: float = 40.0):
        super().__init__()

        pts = np.array(points)
        points_x = pts[:, 0]
        points_y = pts[:, 1]

        new_points_array = np.vstack((points_x, points_y), dtype=np.float64)
        result = scipy.interpolate.splprep(new_points_array, s=.25)
        self.__tck = result[0]
        self.__u = result[1]

        self.__update_measure()

        self._origin = self._barycenter
        self._origin_is_center = True
        self._closure_step = self._perimeter / divider
        self._max_closure_step = self._perimeter / 10
        self._min_closure_step = self._perimeter / 1000

    def __update_measure(self):
        high_definition_u = np.linspace(0, 1, 1000)
        high_definition_coordinate = scipy.interpolate.splev(high_definition_u, self.__tck)
        high_definition_closure = np.column_stack((high_definition_coordinate[0], high_definition_coordinate[1]))
        print(high_definition_closure)

        self._perimeter = Shape.calc_perimeter(high_definition_closure)
        self._barycenter = Shape.calc_barycenter(high_definition_closure)
        self._area = Shape.calc_area(high_definition_closure)
        self._boundary = Shape.calc_boundary(high_definition_closure)

    def point_at(self, t: np.typing.ArrayLike) -> np.typing.NDArray[np.float64]:
        x, y = scipy.interpolate.splev(t, self.__tck)
        return np.column_stack((x, y))
    @property
    def t_range(self) -> tuple[float, float]:
        return 0, 1.0

    def _discretization(self):

        pass'''