from typing import Self

import numpy as np
from numpy import typing as npt

import warnings

from scipy.interpolate import splprep, splev, interp1d
from pygarp.core.models.shapes.parametric_interface import ParametricShape
from pygarp.core.models.commons import (
    Eps,
    ArrayLike,
    TransformationRef,
    EpsConfig,
    DiscretizationMethod,
)
from pygarp.core.models.validators import (
    validate_array_of_nd_coordinates,
    validate_nd_coordinates,
)


class ClosedSpline(ParametricShape):
    standard_definition = 10**ParametricShape.shape_order

    def __init__(
        self,
        control_points: ArrayLike,
        smoothness: float = 0.25,
        definition: int = standard_definition,
        *,
        origin: ArrayLike = None,
        identifier: str | None = None,
        name: str | None = None,
        description: str | None = None,
        _skip: bool = False,
        eps: EpsConfig | float = Eps.eps10,
    ):
        if _skip:
            control_points_array = validate_array_of_nd_coordinates(control_points, 2)
            origin_array = (
                None if origin is None else validate_nd_coordinates(origin, 2)
            )
        else:
            control_points_array = control_points
            origin_array = None if origin is None else origin

        spline_data = splprep(
            control_points_array.astype(np.float64).T, s=smoothness, per=1
        )

        self._tck = spline_data[0]
        self._u = spline_data[1]

        hdu = np.linspace(0, 1, definition)
        self._high_definition_u = hdu
        hdx, hdy = splev(hdu, self._tck)
        self._high_definition_closure = np.column_stack((hdx, hdy))

        super().__init__(
            points=self._high_definition_closure,
            origin=origin_array,
            identifier=identifier,
            name=name,
            description=description,
            _skip=True,
            assume_sort=True,
            eps=eps,
        )

    @property
    def high_definition_u(self) -> npt.NDArray[np.float64]:
        return self._high_definition_u

    @property
    def high_definition_closure(self) -> npt.NDArray[np.float64]:
        return self._high_definition_closure

    def point_at(self, t: ArrayLike) -> npt.NDArray[np.float64]:
        t_array = np.atleast_1d(t)
        t_array = t_array % 1.0
        x, y = splev(t_array, self._tck)
        return np.column_stack((x, y))

    @property
    def t_range(self) -> tuple[float, float]:
        return 0.0, 1.0

    def discretize(
        self,
        *,
        discretization_method: DiscretizationMethod = None,
        custom_step: float | None = None,
    ) -> npt.NDArray[np.float64]:
        discretization_method = (
            discretization_method
            if discretization_method is None or discretization_method == "uniform"
            else None
        )
        return super().discretize(
            discretization_method=discretization_method, custom_step=custom_step
        )

    def _discretization_uniform(self):
        if self.step is None:
            self.step = self.length / 100.0

        x_fine = self.high_definition_closure[:, 0]
        y_fine = self.high_definition_closure[:, 1]

        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        segment_distances = np.sqrt(dx**2 + dy**2)
        cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0)

        u_from_dist = interp1d(cumulative_distance, self._high_definition_u)

        tot_distance = cumulative_distance[-1]
        num_of_segments = max(1, int(np.round(tot_distance / self.step)))
        num_of_points = num_of_segments + 1

        distanze_target = np.linspace(0, tot_distance, num_of_points)
        u_equidistanti = u_from_dist(distanze_target)

        x_equi, y_equi = splev(u_equidistanti, self._tck)
        self._closure = np.column_stack((x_equi, y_equi))

        return self._closure

    @property
    def sure_steps(self) -> npt.NDArray[np.float64] | None:
        warnings.warn(f"Step sicuri non definiti per le spline")
        return None

    def set_step(
        self,
        step: float,
        *,
        cast: bool = True,
        eps: EpsConfig | float = Eps.eps10,
        warn: bool = False,
    ):
        step = float(abs(step))

        if step < self.min_step - eps:
            if warn:
                warnings.warn(
                    f"Step al di fuori dei limiti (troppo piccolo), cast a {self.min_step}"
                )
            self._step = self.min_step
            return

        if step > self.max_step + eps:
            if warn:
                warnings.warn(
                    f"Step al di fuori dei limiti (troppo grande), cast a {self.max_step}"
                )
            self._step = self.max_step
            return

        self._step = step

    def _calc_step_max(self) -> float:
        max_discretization_step = self.length / 3
        return max_discretization_step

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> Self:
        super().translate(x_off, y_off)

        t, c, k = self._tck
        new_c = [c[0] + x_off, c[1] + y_off]
        self._tck = (t, new_c, k)

        self._high_definition_closure += [x_off, y_off]
        return self

    def rotate(self, angle: float = 0.0, ref: TransformationRef = "origin") -> Self:
        if ref not in ("origin", "center"):
            raise ValueError(f"riferimento per la rotazione non valido")

        super().rotate(angle, ref)

        ref_point = self.origin if ref == "origin" else self.barycenter
        cx, cy = ref_point[0], ref_point[1]
        theta = np.radians(angle)
        cos_val = np.cos(theta)
        sin_val = np.sin(theta)

        t, c, k = self._tck
        cx_shifted = c[0] - cx
        cy_shifted = c[1] - cy
        new_c_x = cx_shifted * cos_val - cy_shifted * sin_val + cx
        new_c_y = cx_shifted * sin_val + cy_shifted * cos_val + cy
        self._tck = (t, [new_c_x, new_c_y], k)

        hd_shifted = self._high_definition_closure - [cx, cy]

        rotation_matrix = np.array([[cos_val, sin_val], [-sin_val, cos_val]])

        self._high_definition_closure = np.dot(hd_shifted, rotation_matrix) + [cx, cy]
        return self

    def scale(
        self,
        x_fact: float = 1.0,
        y_fact: float = 1.0,
        ref: TransformationRef = "origin",
    ) -> Self:
        super().scale(x_fact, y_fact, ref)

        ref_origin = self.origin if ref == "origin" else self.barycenter
        cx, cy = ref_origin[0], ref_origin[1]

        t, c, k = self._tck
        new_c_x = cx + (c[0] - cx) * x_fact
        new_c_y = cy + (c[1] - cy) * y_fact
        self._tck = (t, [new_c_x, new_c_y], k)

        hd_shifted = self._high_definition_closure - [cx, cy]
        hd_scaled = hd_shifted * [x_fact, y_fact]

        self._high_definition_closure = hd_scaled + [cx, cy]
        return self
