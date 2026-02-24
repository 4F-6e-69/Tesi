import warnings
import numpy as np
from numpy import typing as npt
from src.obj.shape.parametric_curve import ParametricCurve
from src.obj.shape.shape import Shape

class Circle(ParametricCurve):
    def __init__(self, radius: float, center: np.typing.ArrayLike[np.float64] = (0, 0), step: float = 0.1):
        if radius <= 0:
            radius = abs(float(radius))
            warnings.warn("Il raggio deve essere strettamente positivo.")

        super().__init__(origin=center, step=step)

        self._radius = float(radius)
        self._center = self.origin.copy()
        self._origin_is_center = True


    @property
    def radius(self) -> float:
        return self._radius
    @radius.setter
    def radius(self, value: float):
        if value <= 0:
            raise ValueError("Il raggio deve essere positivo.")
        self._radius = float(value)
        self._closure = None
        self.reset_cache()

    @property
    def center(self) -> npt.NDArray[np.float64]:
        return self._center

    @center.setter
    def center(self, value: np.typing.ArrayLike[np.float64]):
        new_c = np.array(value, dtype=np.float64)

        if new_c.shape != (2,):
            raise ValueError("Il centro deve essere un array (2,)")

        if np.allclose(new_c, self._center, atol=self.tolerance):
            return

        self._center = new_c
        self._origin_is_center = np.allclose(self.origin, new_c, atol=self.tolerance)

        self._closure = None
        self.reset(["boundary"])

    @property
    def area(self) -> float:
        return np.pi * (self.radius ** 2)
    @property
    def perimeter(self) -> float:
        return 2 * np.pi * self.radius
    @property
    def barycenter(self) -> npt.NDArray[np.float64]:
        return self._center
    @property
    def boundary(self) -> tuple[float, float, float, float, float, float] | None:
        if self._boundary is None:
            if self._center is None or self._radius is None:
                raise ValueError("Impossibile calcolo del boundary")
            self._boundary = (self._center[0] - self._radius,
                              self._center[0] + self._radius,
                              self._center[1] - self._radius,
                              self._center[1] + self._radius,
                              0.0, 0.0)

        return self._boundary

    @property
    def t_range(self) -> tuple[float, float]:
        return 0.0, 2 * np.pi
    @property
    def is_closed_loop(self) -> bool:
        return True
    def point_at(self, t: np.typing.ArrayLike[np.float64]) -> np.typing.NDArray[np.float64]:
        x = self.radius * np.cos(t)
        y = self.radius * np.sin(t)

        if np.isscalar(t):
            return self.center + np.array([x, y])

        return self.center + np.column_stack((x, y))

    def translate(self, offset: np.typing.ArrayLike[np.float64]):
        delta = np.asarray(offset, dtype=np.float64)
        if delta.shape != (2,):
            raise ValueError("L'offset deve essere (dx, dy)")

        self._center += delta
        self._origin += delta

        self._closure = None
        self.reset(["boundary"])
    def rotate(self, angle_rad: float, ref_center: bool = True):
        if np.isclose(angle_rad, 0.0, atol=self.tolerance):
            return

        if ref_center:
            return

        ref = self.origin
        self._center = Shape._rotate_points(angle_rad, self._center, ref)
        self._closure = None
        self.reset(["boundary"])
    def scale(self, factors: np.typing.ArrayLike[np.float64], ref_center: bool = True):
        s = np.asarray(factors, dtype=np.float64)
        if s.ndim == 0:
            sx = sy = float(s)
        else:
            sx, sy = s[0], s[1]

        if not np.isclose(sx, sy):
            raise ValueError(
                "Impossibile scalare un Cerchio in modo non uniforme (diventerebbe un'Ellisse). Usa fattori uguali.")

        if np.isclose(sx, 1.0, atol=self.tolerance):
            return

        self._radius *= sx
        if not ref_center:
            vec = self._center - self.origin
            self._center = self.origin + (vec * sx)

        self._closure = None
        self.reset_cache()