import numpy as np
import scipy
from src.obj.shape.parametric_curve import ParametricCurve
from src.obj.shape.shape import Shape

class Spline(ParametricCurve):
    def __init__(self, points: np.typing.ArrayLike[float], divider: float = 40.0):
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

        Shape.calc_barycenter(self._closure)

    def __update_measure(self):
        high_definition_u = np.linspace(0, 1, 1000)
        high_definition_closure = scipy.interpolate.splev(high_definition_u, self.__tck)

        self._perimeter = Shape.calc_perimeter(high_definition_closure)
        self._barycenter = Shape.calc_barycenter(high_definition_closure)
        self._area = Shape.calc_area(high_definition_closure)
        self._boundary = Shape.calc_boundary(high_definition_closure)

    def point_at(self, t: np.typing.ArrayLike[float]) -> np.typing.NDArray[np.float64]:
        x, y = scipy.interpolate.splev(t, self.__tck)
        return np.concatenate((x, y), axis=1)
    @property
    def t_range(self) -> tuple[float, float]:
        return 0, 1.0