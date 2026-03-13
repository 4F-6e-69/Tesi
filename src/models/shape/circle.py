import numpy as np
from numpy import typing as npt

from src.models.shape.parametric_shape import ParametricShape
from src.utils import ArrayLike, validate_2d_coordinates, Ref, Eps


class Circle(ParametricShape):
    def __init__(self, center: ArrayLike, radius: float):
        new_center = validate_2d_coordinates(center)
        c = np.asarray(center if new_center is None else new_center, dtype=np.float64)

        self._center: npt.NDArray[np.float64] = c
        self._radius: float = float(radius)

        __vertices_matrix = np.asarray([
            [0, self._radius],
            [self._radius, 0],
            [0, -self._radius],
            [-self._radius, 0],
        ], dtype=np.float64)
        points_array = __vertices_matrix + c

        super().__init__(points=points_array, origin=c, __skip=True)

    @property
    def center(self) -> npt.NDArray[np.float64]:
        return self._center
    @property
    def radius(self) -> float:
        return self._radius

    @property
    def theoretical_area(self) -> float:
        return np.pi * (self.radius ** 2)
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

    def _discretization(self):
        pass

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> 'Circle':
        super().translate(x_off, y_off)
        self._center = self.center + np.array([x_off, y_off], dtype=np.float64)
        return self
    def rotate(self, angle: float = 0.0, ref: Ref = "origin") -> 'Circle':
        if ref == "barycenter":
            super().rotate(angle, ref)
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

        raise ValueError(f"Riferimento: {ref} non valido REF: [origin, barycenter, center]")
    def scale(self, scale: float = 1.0, ref: Ref = "origin", **kwargs) -> 'Circle':
        if ref == "origin":
            ref_pt = self.origin
        elif ref == "barycenter":
            ref_pt = self.barycenter
        else:
            raise ValueError(f"Riferimento: {ref} non valido REF: [origin, barycenter, center]")

        super().scale(x_fact=scale, y_fact=scale, ref=ref)

        self._radius = self._radius * abs(scale)
        self._center = ref_pt + (self._center - ref_pt) * scale
        return self