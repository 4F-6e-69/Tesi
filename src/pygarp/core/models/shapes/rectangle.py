import warnings
from typing import Self

import numpy as np
from numpy import typing as npt
from shapely.geometry import Polygon

from pygarp.core.models.shapes.shape.core import Shape
from pygarp.core.models.commons import ArrayLike, Eps, EpsConfig, TransformationRef
from pygarp.core.models.validators import validate_nd_coordinates


class Rectangle(Shape):
    def __init__(
            self,
            width: float,
            height: float,
            center: ArrayLike | None = None,
            *,
            origin: ArrayLike = None,
            identifier: str | None = None,
            name: str | None = None,
            description: str | None = None,
            _skip: bool = False,
            eps: EpsConfig | float = Eps.eps10,
    ):
        if _skip:
            w_float = width
            h_float = height
            center_array = np.zeros(2, dtype=np.float64) if center is None else center
            origin_array = origin
        else:
            w_float = float(abs(width))
            h_float = float(abs(height))

            if w_float < eps or h_float < eps:
                raise ValueError("Larghezza e altezza devono essere maggiori di zero.")

            center_array = np.zeros(2, dtype=np.float64) if center is None else validate_nd_coordinates(center, 2)
            origin_array = np.zeros(2, dtype=np.float64) if origin is None else validate_nd_coordinates(origin, 2)

        self._width = w_float
        self._height = h_float
        self._center = center_array
        self._angle = 0.0

        vertices = Rectangle._compone_rectangle(self._center, self._width, self._height, self._angle)

        super().__init__(
            points=vertices,
            origin=origin_array,
            identifier=identifier,
            name=name,
            description=description,
            _skip=True,
            assume_sort=True,  # I vertici generati sono già ordinati in senso antiorario
            eps=eps,
        )

    @staticmethod
    def _compone_rectangle(center: npt.NDArray[np.float64], w: float, h: float, angle: float) -> npt.NDArray[
        np.float64]:
        cx, cy = center[0], center[1]
        half_w, half_h = w / 2.0, h / 2.0

        vertices = np.array([
            [cx - half_w, cy - half_h],  # Bottom-Left
            [cx + half_w, cy - half_h],  # Bottom-Right
            [cx + half_w, cy + half_h],  # Top-Right
            [cx - half_w, cy + half_h]  # Top-Left
        ], dtype=np.float64)

        if angle != 0.0:
            theta = np.radians(angle)
            cos_val, sin_val = np.cos(theta), np.sin(theta)

            shifted = vertices - center
            rotated = np.dot(shifted, np.array([[cos_val, sin_val], [-sin_val, cos_val]]))
            vertices = rotated + center

        return vertices

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = max(float(abs(value)), Eps.eps10)
        vertices = Rectangle._compone_rectangle(self._center, self._width, self._height, self._angle)
        super()._set_polygon(Polygon(vertices))

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float):
        self._height = max(float(abs(value)), Eps.eps10)
        vertices = Rectangle._compone_rectangle(self._center, self._width, self._height, self._angle)
        super()._set_polygon(Polygon(vertices))

    @property
    def center(self) -> npt.NDArray[np.float64]:
        if self._center is None:
            self._center = np.array([self.polygon.centroid.x, self.polygon.centroid.y], dtype=np.float64)
        return self._center

    @center.setter
    def center(self, center: ArrayLike):
        self._center = validate_nd_coordinates(center, 2)
        vertices = Rectangle._compone_rectangle(self._center, self._width, self._height, self._angle)
        super()._set_polygon(Polygon(vertices))

    @property
    def theoretical_area(self) -> float:
        return self._width * self._height

    @property
    def theoretical_perimeter(self) -> float:
        return 2 * (self._width + self._height)

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> Self:
        super().translate(x_off, y_off)
        self._center = self._center + np.array([x_off, y_off], dtype=np.float64)
        return self

    def rotate(self, angle: float = 0.0, ref: TransformationRef = "origin") -> Self:
        if ref not in ("origin", "barycenter", "center"):
            raise ValueError(f"Riferimento {ref} non valido.")

        super().rotate(angle, ref)

        self._angle = (self._angle + angle) % 360.0

        if ref in ("barycenter", "center"):
            return self

        theta = np.radians(angle)
        cos_val, sin_val = np.cos(theta), np.sin(theta)
        cx, cy = self.center[0] - self.origin[0], self.center[1] - self.origin[1]
        new_cx = cx * cos_val - cy * sin_val + self.origin[0]
        new_cy = cx * sin_val + cy * cos_val + self.origin[1]

        self._center = np.array([new_cx, new_cy], dtype=np.float64)
        return self

    def scale(self, x_fact: float = 1.0, y_fact: float = 1.0, ref: TransformationRef = "origin") -> Self:
        if ref not in ("origin", "barycenter", "center"):
            raise ValueError(f"Riferimento {ref} non valido.")

        super().scale(x_fact, y_fact, ref)

        if self._angle % 90.0 != 0 and abs(x_fact - y_fact) > Eps.eps10:
            warnings.warn(
                "Scaling non uniforme su un rettangolo ruotato: la forma risultante non sarà più un rettangolo esatto.")

        self._width *= abs(x_fact)
        self._height *= abs(y_fact)

        ref_point = self.origin if ref == "origin" else self.barycenter
        self._center = ref_point + (self._center - ref_point) * x_fact

        return self