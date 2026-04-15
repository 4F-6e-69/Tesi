from typing import Self
import warnings

import math
import numpy as np

from pygarp.core.models.shapes.shape.core import Shape
from pygarp.core.models.commons import ArrayLike, Eps, EpsConfig, TransformationRef
from pygarp.core.models.validators import validate_nd_coordinates


class RegularPolygon(Shape):
    def __init__(
        self,
        side: float,
        num_of_sides: int,
        center: ArrayLike,
        *,
        origin: ArrayLike = None,
        identifier: str | None = None,
        name: str | None = None,
        description: str | None = None,
        _skip: bool = False,
        eps: EpsConfig | float = Eps.eps10,
    ):
        if _skip:
            n = num_of_sides
            side_float = side
            center_array = np.zeros(2, dtype=np.float64) if center is None else center
            origin_array = origin
        else:
            n = math.floor(num_of_sides)
            if n < 3:
                raise ValueError("Un poligono deve avere almeno 3 lati.")
            side_float = float(abs(side))
            center_array = (
                np.zeros(2, dtype=np.float64)
                if center is None
                else validate_nd_coordinates(center, 2)
            )
            origin_array = (
                np.zeros(2, dtype=np.float64)
                if origin is None
                else validate_nd_coordinates(origin, 2)
            )

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = side_float / (2 * math.sin(np.pi / n))
        vertices = np.column_stack([np.cos(angles), np.sin(angles)]) * r + center_array

        self._side = side_float
        self._n = n
        self._center = center_array

        self._apothem = side_float / (2 * math.tan(np.pi / n))

        super().__init__(
            points=vertices,
            origin=origin_array,
            identifier=identifier,
            name=name,
            description=description,
            _skip=True,
            assume_sort=True,
            eps=eps,
        )

    @property
    def side(self) -> float:
        return self._side

    @property
    def num_of_sides(self) -> int:
        return self._n

    @property
    def center(self) -> ArrayLike:
        if self._center is None:
            self._center = np.array(
                [self.polygon.centroid.x, self.polygon.centroid.y], dtype=np.float64
            )
        return self._center

    @property
    def apothem(self) -> float:
        return self._apothem

    @property
    def vertices(self):
        return self.control_points

    @property
    def theoretical_area(self) -> float:
        return self.side * self.num_of_sides * self.apothem / 2

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> Self:
        super().translate(x_off, y_off)
        self._center = self._center + np.array([x_off, y_off], dtype=np.float64)
        return self

    def rotate(self, angle: float = 0.0, ref: TransformationRef = "origin") -> Self:
        if ref not in ("origin", "barycenter", "center"):
            raise ValueError(f"Riferimento {ref} non valido.")

        if ref in ("barycenter", "center"):
            # Ruotare un poligono regolare sul proprio centro non sposta il centro
            super().rotate(angle, ref)
            return self

        super().rotate(angle, ref)
        theta = np.radians(angle)
        cos_val, sin_val = np.cos(theta), np.sin(theta)
        cx, cy = self.center[0] - self.origin[0], self.center[1] - self.origin[1]
        new_cx = cx * cos_val - cy * sin_val + self.origin[0]
        new_cy = cx * sin_val + cy * cos_val + self.origin[1]
        self._center = np.array([new_cx, new_cy], dtype=np.float64)
        return self

    def scale(
        self,
        x_fact: float = 1.0,
        y_fact: float = 1.0,
        ref: TransformationRef = "origin",
    ) -> Self:
        if ref not in ("origin", "barycenter", "center"):
            raise ValueError(f"Riferimento {ref} non valido.")

        if abs(x_fact - y_fact) > Eps.eps10:
            warnings.warn(
                "Scala non uniforme: il poligono non sarà più regolare! Le proprietà base verranno forzate su x_fact."
            )

        super().scale(x_fact, y_fact, ref)

        self._side *= abs(x_fact)
        self._apothem *= abs(x_fact)

        ref_point = self.origin if ref == "origin" else self.barycenter
        self._center = ref_point + (self._center - ref_point) * x_fact
        return self
