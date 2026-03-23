import warnings
from typing import Self

import numpy as np
from numpy import typing as npt
from shapely import Point, Polygon

from pygarp.core.models.commons import ArrayLike, EpsConfig, Eps, TransformationRef
from pygarp.core.models.shapes.parametric_interface import ParametricShape
from pygarp.core.models.validators import validate_nd_coordinates


class Circle(ParametricShape):
    def __init__(
        self,
        radius: float,
        center: ArrayLike = None,
        *,
        origin: ArrayLike = None,
        identifier: str | None = None,
        name: str | None = None,
        description: str | None = None,
        _skip: bool = False,
        eps: EpsConfig | float = Eps.eps10,
    ):
        if not _skip:
            center_array = validate_nd_coordinates(center, 2)
            origin_array = (
                validate_nd_coordinates(origin, 2) if origin is not None else None
            )
            radius_float = float(abs(radius))
            if radius_float < eps:
                raise ValueError("Raggio del cerchio troppo piccolo")
        else:
            center_array = center
            origin_array = origin if origin is not None else None
            radius_float = radius

        self._center: npt.NDArray[np.float64] = center_array
        self._radius: float = radius_float
        cerchio: Polygon = Point(self._center).buffer(self._radius)

        super().__init__(
            points=np.asarray(
                np.column_stack(cerchio.exterior.coords.xy), dtype=np.float64
            ),
            origin=origin_array,
            identifier=identifier,
            name=name,
            description=description,
            _skip=True,
            assume_sort=True,
            eps=eps,
        )

    @property
    def center(self) -> npt.NDArray[np.float64]:
        return self._center

    @center.setter
    def center(self, center: npt.NDArray[np.float64]):
        self._center = validate_nd_coordinates(center, 2)
        super()._set_polygon(Point(self._center).buffer(self._radius))

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, radius: float):
        radius_float = float(abs(radius))
        self._radius = radius_float if radius_float > Eps.eps10 else Eps.eps10
        super()._set_polygon(Point(self._center).buffer(self._radius))

    @property
    def theoretical_area(self) -> float:
        return np.pi * (self.radius**2)

    @property
    def theoretical_circumference(self) -> float:
        return 2 * np.pi * self.radius

    def point_at(self, t: ArrayLike) -> npt.NDArray[np.float64]:
        t_array = np.atleast_1d(t)

        x = self.radius * np.cos(t_array) + self.center[0]
        y = self.radius * np.sin(t_array) + self.center[1]
        return np.column_stack((x, y))

    @property
    def t_range(self) -> tuple[float, float]:
        return 0.0, 2 * np.pi

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> Self:
        super().translate(x_off, y_off)
        self._center = self.center + np.array([x_off, y_off], dtype=np.float64)
        return self

    def rotate(self, angle: float = 0.0, ref: TransformationRef = "origin") -> Self:
        if ref == "center":
            return self

        if ref == "origin":
            super().rotate(angle, ref)

            theta = np.radians(angle)
            cos_val, sin_val = np.cos(theta), np.sin(theta)

            cx, cy = self.center[0] - self.origin[0], self.center[1] - self.origin[1]
            new_cx = cx * cos_val - cy * sin_val + self.origin[0]
            new_cy = cx * sin_val + cy * cos_val + self.origin[1]

            self._center = np.array([new_cx, new_cy], dtype=np.float64)
            return self

        raise ValueError(f"Riferimento: {ref} non valido REF: [origin, center]")

    def scale(
        self,
        x_fact: float = 1.0,
        y_fact: float = 1.0,
        ref: TransformationRef = "origin",
    ) -> Self:
        if ref != "origin" and ref != "center":
            raise ValueError(f"Riferimento: {ref} non valido REF: [origin, center]")

        if abs(x_fact) < Eps.eps10:
            raise ValueError(f"Impossible fare lo scaling per 0.0")

        if abs(x_fact - y_fact) > Eps.eps10:
            warnings.warn(
                f"x_fact {x_fact} e y_fact {y_fact} troppo diversi, scala non uniforme, cast del fattore di scala a {x_fact}"
            )

        super().scale(x_fact=x_fact, y_fact=x_fact, ref=ref)

        self._radius = self._radius * abs(x_fact)
        ref_point = self.origin if ref == "origin" else self.barycenter
        self._center = ref_point + (self._center - ref_point) * x_fact
        return self
