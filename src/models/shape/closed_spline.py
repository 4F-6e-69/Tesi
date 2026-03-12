import numpy as np
from numpy import typing as nptyping

from scipy.interpolate import splprep, splev, interp1d
from src.models.shape.parametric_shape import ParametricShape
from src.utils import ArrayLike, Ref
from src.utils import validate_array_of_2d_coordinates


class ClosedSpline(ParametricShape):
    standard_definition = 10 ** ParametricShape.shape_order

    def __init__(self, control_points: ArrayLike, smoothness: float = 0.25, definition: int = standard_definition):
        # validazione
        new_control_points_array = validate_array_of_2d_coordinates(control_points)
        if new_control_points_array is None:
            new_control_points_array = np.asarray(control_points)

        # calcolo della Spline
        spline_data = splprep(new_control_points_array.astype(np.float64).T, s=smoothness, per=1)

        self.__tck = spline_data[0]
        self.__u = spline_data[1]

        # generazione dei punti ad alta definizione
        hdu = np.linspace(0, 1, definition)
        self._high_definition_u = hdu
        hdx, hdy = splev(hdu, self.__tck)
        self._high_definition_closure = np.column_stack((hdx, hdy))

        # inizializzazione della super-classe
        super().__init__(
            points=self._high_definition_closure,
            origin=np.zeros(2, dtype=np.float64),
            __skip=True
        )

    @property
    def high_definition_u(self) -> nptyping.NDArray[np.float64]:
        return self._high_definition_u
    @property
    def high_definition_closure(self) -> nptyping.NDArray[np.float64]:
        return self._high_definition_closure

    def point_at(self, t: ArrayLike) -> nptyping.NDArray[np.float64]:
        t_array = np.atleast_1d(t)
        t_array = t_array % 1.0
        x, y = splev(t_array, self.__tck)
        return np.column_stack((x, y))
    @property
    def t_range(self) -> tuple[float, float]:
        return 0.0, 1.0

    def _discretization(self):
        # 1. Fallback specifico per le Spline (spostato qui!)
        if self.discretization_step is None:
            self.discretization_step = self.length / 100.0

        # 2. Riparametrizzazione per lunghezza d'arco
        x_fine = self.high_definition_closure[:, 0]
        y_fine = self.high_definition_closure[:, 1]

        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        segment_distances = np.sqrt(dx ** 2 + dy ** 2)
        cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0)

        u_from_dist = interp1d(cumulative_distance, self._high_definition_u)

        # 3. Calcolo punti adattivo
        tot_distance = cumulative_distance[-1]
        num_of_segments = max(1, int(np.round(tot_distance / self.discretization_step)))
        num_of_points = num_of_segments + 1

        distanze_target = np.linspace(0, tot_distance, num_of_points)
        u_equidistanti = u_from_dist(distanze_target)

        # 4. Generazione coordinate
        x_equi, y_equi = splev(u_equidistanti, self.__tck)
        self._closure = np.column_stack((x_equi, y_equi))

        return self._closure

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> 'ClosedSpline':
        super().translate(x_off, y_off)

        t, c, k = self.__tck
        new_c = [c[0] + x_off, c[1] + y_off]
        self.__tck = (t, new_c, k)

        self._high_definition_closure += [x_off, y_off]
        return self
    def rotate(self, angle: float = 0.0, ref: Ref = "origin") -> 'ClosedSpline':
        ref_origin = self.origin if ref == "origin" else self.barycenter
        cx, cy = ref_origin[0], ref_origin[1]

        super().rotate(angle, ref)

        theta = np.radians(angle)
        cos_val = np.cos(theta)
        sin_val = np.sin(theta)

        t, c, k = self.__tck
        cx_shifted = c[0] - cx
        cy_shifted = c[1] - cy
        new_c_x = cx_shifted * cos_val - cy_shifted * sin_val + cx
        new_c_y = cx_shifted * sin_val + cy_shifted * cos_val + cy
        self.__tck = (t, [new_c_x, new_c_y], k)

        hd_shifted = self._high_definition_closure - [cx, cy]

        rotation_matrix = np.array([
            [cos_val, sin_val],
            [-sin_val, cos_val]
        ])

        self._high_definition_closure = np.dot(hd_shifted, rotation_matrix) + [cx, cy]
        return self
    def scale(self, x_fact: float = 1.0, y_fact: float = 1.0, ref: Ref = "origin") -> 'ClosedSpline':
        ref_origin = self.origin if ref == "origin" else self.barycenter
        cx, cy = ref_origin[0], ref_origin[1]

        super().scale(x_fact, y_fact, ref)

        t, c, k = self.__tck
        new_c_x = cx + (c[0] - cx) * x_fact
        new_c_y = cy + (c[1] - cy) * y_fact
        self.__tck = (t, [new_c_x, new_c_y], k)

        hd_shifted = self._high_definition_closure - [cx, cy]
        hd_scaled = hd_shifted * [x_fact, y_fact]

        self._high_definition_closure = hd_scaled + [cx, cy]
        return self