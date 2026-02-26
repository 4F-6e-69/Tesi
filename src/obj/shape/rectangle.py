import warnings
import numpy as np

from src.obj.shape.polygon import Polygon

class Rectangle(Polygon):
    def __init__(self, origin: np.typing.ArrayLike = (0, 0), origin_is_center: bool = False, width: float = 17.5, height: float = 12.5):
        self._width = float(width)
        self._height = float(height)

        if self._width <= 0 or self._height <= 0:
            raise ValueError("Larghezza e altezza devono essere positive.")

        org = np.asarray(origin, dtype=np.float64)
        dx = self._width
        dy = self._height

        if origin_is_center:
            offsets = np.array([
                [-dx / 2, dy / 2],  # A (Top-Left)
                [dx / 2, dy / 2],  # B (Top-Right)
                [dx / 2, -dy / 2],  # C (Bottom-Right)
                [-dx / 2, -dy / 2]  # D (Bottom-Left)
            ])
            shape_origin = org
        else:
            offsets = np.array([
                [0, dy],  # Top-Left
                [dx, dy],  # Top-Right
                [dx, 0],  # Bottom-Right
                [0, 0]  # Bottom-Left (Origin)
            ])
            shape_origin = org

        final_vertices = org + offsets
        super().__init__(vertices=final_vertices, origin=shape_origin)

    @classmethod
    def new_rectangle(cls, o: np.typing.ArrayLike, a: np.typing.ArrayLike, c: np.typing.ArrayLike):
        w = c[0] - o[0]
        h = a[1] - o[1]

        return cls(origin=o, origin_is_center=False, width=w, height=h)
    @classmethod
    def new_rect(cls, a: np.typing.ArrayLike, c: np.typing.ArrayLike):
        w = np.abs(c[0] - a[0])
        h = np.abs(a[1] - c[1])

        return cls(origin=np.mean(np.asarray([a, c]), axis=0), origin_is_center=True, width=w, height=h)

    @property
    def width(self) -> float:
        return self._width
    @property
    def height(self) -> float:
        return self._height

    @property
    def area(self) -> float:
        if self._area is None:
            self._ensure_valid_vertices("area")
            self._area = self.width * self.height
        return self._area
    @property
    def perimeter(self) -> float:
        if self._perimeter is None:
            self._ensure_valid_vertices("perimeter")
            self._perimeter = 2 * (self.width + self.height)
        return self._perimeter
    @property
    def barycenter(self):
        if self._barycenter is None:
            self._ensure_valid_vertices("baricentro")
            self._barycenter = (self.vertices[0] + self.vertices[2]) / 2
        return self._barycenter

    def scale(self, factor: np.typing.ArrayLike):
        s = np.asarray(factor, dtype=float)

        if s.ndim == 0:
            sx = sy = float(s)
        else:
            sx, sy = s[0], s[1]
            if not np.isclose(sx, sy):
                warnings.warn("Scaling non uniforme su un Rettangolo: se ruotato, larghezza/altezza potrebbero perdere significato geometrico.")

        super().scale(factor)

        self._width *= sx
        self._height *= sy