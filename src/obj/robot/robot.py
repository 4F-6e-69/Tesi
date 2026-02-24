import warnings
import numpy as np
from typing import Tuple
from src.obj.plane.working_plane import WorkingSpace

class Robot:
    def __init__(self):
        self._max_radius: float | None = None
        self._min_radius: float | None = None

        self._abs_tool_alpha: float | None = None
        self._abs_tool_radius: float | None = None

        self._tolerance = WorkingSpace.EPS_12

    @classmethod
    def new_robot(cls, max_radius=None, min_radius=None, abs_tool_alpha=None, abs_tool_radius=None):
        robot = cls()

        robot.max_radius = max_radius
        robot.min_radius = min_radius

        robot.abs_tool_alpha = abs_tool_alpha
        robot.abs_tool_radius = abs_tool_radius

        return robot

    @property
    def max_radius(self) -> float:
        if self._max_radius is None:
            raise AttributeError("Raggio massimo (max_radius) non configurato.")
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float):
        if value < 0:
            warnings.warn(f"Raggio negativo rilevato ({value}). Verrà utilizzato il valore assoluto: {abs(value)}")

        val_float = float(abs(value))

        if self._min_radius is not None and val_float <= self._min_radius:
            raise ValueError(
                f"Errore: Il raggio massimo ({val_float}) deve essere maggiore "
                f"del raggio minimo attuale ({self._min_radius})."
            )

        self._max_radius = val_float

    @property
    def min_radius(self) -> float:
        if self._min_radius is None:
            raise AttributeError("Raggio minimo (min_radius) non configurato.")
        return self._min_radius

    @min_radius.setter
    def min_radius(self, value: float):
        if value < 0:
            warnings.warn(f"Raggio negativo rilevato ({value}). Verrà utilizzato il valore assoluto: {abs(value)}")

        val_float = float(abs(value))

        if self._max_radius is not None and val_float >= self._max_radius:
            raise ValueError(
                f"Errore: Il raggio minimo ({val_float}) deve essere minore "
                f"del raggio massimo attuale ({self._max_radius})."
            )

        self._min_radius = val_float

    @property
    def abs_tool_alpha(self) -> float:
        if self._abs_tool_alpha is None:
            raise AttributeError("L'angolo dell'utensile (abs_tool_alpha) non è ancora stato impostato.")
        return self._abs_tool_alpha

    @abs_tool_alpha.setter
    def abs_tool_alpha(self, alpha: float):
        if alpha is None:
            raise ValueError("L'angolo dell'utensile (alpha) non può essere None.")

        if alpha < 0:
            warnings.warn(f"Angolo utensile negativo rilevato ({alpha}). Verrà convertito in positivo: {abs(alpha)}")

        self._abs_tool_alpha = float(abs(alpha))

    @property
    def abs_tool_radius(self) -> float:
        if self._abs_tool_radius is None:
            raise AttributeError("Il raggio dell'utensile (abs_tool_radius) non è ancora stato impostato.")
        return self._abs_tool_radius

    @abs_tool_radius.setter
    def abs_tool_radius(self, value: float):
        if value is None:
            raise ValueError("Il raggio dell'utensile non può essere None.")

        if value < 0:
            warnings.warn(f"Raggio utensile negativo rilevato ({value}). Verrà convertito in positivo: {abs(value)}")

        self._abs_tool_radius = float(abs(value))

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        if value is None:
            raise ValueError("La tolleranza non può essere None.")

        if value <= 0:
            warnings.warn(f"Impostata tolleranza non strettamente positiva ({value}). Verrà usato il valore assoluto.")

        float_value = float(abs(value))
        if float_value > WorkingSpace.EPS_05:
            raise ValueError(f"Tolleranza troppo alta (tolleranza massima = {WorkingSpace.EPS_05}).")

        self._tolerance = float_value

    def validate_point(self, point: np.typing.ArrayLike[np.float64]) -> bool:
        p = np.asarray(point, dtype=np.float64)
        p_norm = np.linalg.norm(p)

        min_bound = self.min_radius - self.tolerance
        max_bound = self.max_radius + self.tolerance

        return min_bound <= p_norm <= max_bound

    def validate_bounding_box(self, bounding: Tuple[float, float, float, float], center_z: float) -> bool:
        if not self.is_valid_model():
            raise RuntimeError("Il modello del robot non è configurato correttamente.")

        min_x, min_y, max_x, max_y = bounding

        corners_x = np.array([min_x, max_x, min_x, max_x], dtype=np.float64)
        corners_y = np.array([min_y, min_y, max_y, max_y], dtype=np.float64)

        max_corner_dist = np.max(np.sqrt(corners_x ** 2 + corners_y ** 2 + center_z ** 2))

        if max_corner_dist > (self.max_radius + self.tolerance):
            return False

        closest_x = np.clip(0.0, min_x, max_x)
        closest_y = np.clip(0.0, min_y, max_y)

        min_dist = np.sqrt(closest_x ** 2 + closest_y ** 2 + center_z ** 2)

        if min_dist < (self.min_radius - self.tolerance):
            return False

        return True

    #TODO not implemented
    def validate_shape_angles(self, angels: tuple[float, float]):
        print(f"angels: {angels}, robot tool width: {self.abs_tool_radius}, robot tool angle: {self.abs_tool_alpha}")
        return NotImplemented

    def is_valid_model(self) -> bool:
        if None in (self._min_radius, self._max_radius, self._abs_tool_radius, self._abs_tool_alpha, self._tolerance):
            return False

        if self._max_radius - self._min_radius <= self._tolerance:
            return False

        if self._abs_tool_radius <= self._tolerance or self._abs_tool_alpha <= self._tolerance:
            return False

        return True

    @staticmethod
    def validate_model(robot_instance) -> bool:
        if not isinstance(robot_instance, Robot):
            raise TypeError("Errore: l'oggetto fornito non è un'istanza della classe Robot.")

        return robot_instance.is_valid_model()