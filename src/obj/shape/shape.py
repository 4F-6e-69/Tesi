import warnings
from abc import ABC, abstractmethod

import numpy as np
from typing import Optional, List, Tuple, Literal
from matplotlib import pyplot as plt

from src.obj.plane.working_plane import WorkingSpace

class Shape(ABC):
    def __init__(self):
        self._origin: np.typing.NDArray[np.float64] = np.zeros(2)
        self._origin_is_center: bool = False

        self._tolerance: float = 1e-6

        self._barycenter: np.typing.NDArray[np.float64] | None = None
        self._perimeter: float | None = None
        self._area: float | None = None
        self._boundary: Tuple[float, float, float, float, float, float] | None = None

        self._closure: np.typing.NDArray[np.float64] | None = None
        self._closure_step: float | None = None
        self._min_closure_step: float = 0.1
        self._max_closure_step: float = 10.0

    @property
    def tolerance(self) -> float:
        return self._tolerance
    @tolerance.setter
    def tolerance(self, value: float | int):
        if value is None:
            raise ValueError("La tolleranza non può essere None.")

        if isinstance(value, bool):
            raise TypeError("Il tipo bool non è ammesso per la tolleranza.")

        final_tolerance = WorkingSpace.EPS_16
        if isinstance(value, int):
            precision = value
            if precision <= 0:
                raise ValueError("La precisione (int) deve essere positiva (es. 6 per 1e-6).")

            new_tolerance = 10.0 ** -precision
            eps64 = np.finfo(np.float64).eps
            if new_tolerance < eps64:
                raise ValueError(f"Precisione troppo alta: valore risultante < epsilon macchina ({eps64})")

            final_tolerance = new_tolerance

        elif isinstance(value, float):
            if value <= 0:
                warnings.warn(f"Tolleranza negativa o nulla ({value}). Verrà usato il valore assoluto.")

            float_value = abs(float(value))
            if float_value > WorkingSpace.EPS_05:
                raise ValueError(f"Tolleranza troppo alta > {WorkingSpace.EPS_05}.")

            eps64 = np.finfo(np.float64).eps
            if float_value < eps64:
                warnings.warn("Tolleranza inferiore all'epsilon macchina, potrebbe non avere effetto.")

            final_tolerance = float_value
        else:
            raise TypeError(f"Tipo non supportato: {type(value)}")

        if not np.isclose(self._tolerance, final_tolerance, atol=1e-15):
            self._tolerance = final_tolerance

    @property
    def origin(self) -> np.typing.NDArray[np.float64]:
        return self._origin
    def _ensure_origin(self, new_origin: np.typing.ArrayLike) -> np.typing.NDArray[np.float64]:
        no = np.asarray(new_origin, dtype=np.float64, copy=True).flatten()
        if no.shape != self._origin.shape:
            raise ValueError(f"Dimensione errata: attesa {self._origin.shape}, ricevuta {no.shape}")

        return no
    @property
    def origin_is_center(self) -> bool:
        return self._origin_is_center
    @origin.setter
    def origin(self, origin: np.typing.ArrayLike):
        new_o = self._ensure_origin(origin)

        self._origin = new_o
        self._set_origin_is_center()
    def _set_origin_is_center(self):
        try:
            if self.closure_is_valid:
                center = self.barycenter
                self._origin_is_center = np.allclose(self.origin, center, atol=self.tolerance)
            else:
                self._origin_is_center = False
        except ValueError:
            self._origin_is_center = False

    @property
    def perimeter(self) -> float | None:
        if self._perimeter is None:
            self._ensure_valid_closure("perimetro")
            self._perimeter = Shape.calc_perimeter(self.closure)
        return self._perimeter
    @property
    def area(self) -> float:
        if self._area is None:
            self._ensure_valid_closure("area")
            self._area = Shape.calc_area(self.closure)
        return abs(self._area)
    @property
    def sign_area(self) -> float:
        if self._area is None:
            self._ensure_valid_closure("area con segno")
            self._area = Shape.calc_area(self.closure)
        return self._area
    @property
    def boundary(self) -> tuple[float, float, float, float, float, float] | None:
        if self._boundary is None:
            self._ensure_valid_closure("boundary")
            self._boundary = Shape.calc_boundary(self.closure)
        return self._boundary
    @property
    def barycenter(self) -> np.typing.NDArray[np.float64]:
        if self._barycenter is None:
            self._ensure_valid_closure("baricentro")
            self._barycenter = Shape.calc_barycenter(self.closure)
        return self._barycenter

    @property
    def closure_is_valid(self) -> bool:
        return (self._closure is not None and
                isinstance(self._closure, np.ndarray) and
                self._closure.shape[0] >= 3)
    def _ensure_valid_closure(self, property_name: str):
        if not self.closure_is_valid:
            n_points = len(self._closure) if self._closure is not None else 0
            raise ValueError(f"Impossibile calcolare {property_name}: servono almeno 3 punti (trovati: {n_points}).")
    @property
    def closure(self) -> np.typing.NDArray[np.float64] | None:
        return self._closure
    @property
    def closure_step(self) -> float | None:
        return self._closure_step
    def validate_step(self, step: float) -> bool:
        return self.min_closure_step - self.tolerance < step < self.max_closure_step + self.tolerance
    @property
    def min_closure_step(self) -> float | None:
        return self._min_closure_step
    @property
    def max_closure_step(self) -> float | None:
        return self._max_closure_step

    @closure_step.setter
    def closure_step(self, value: float):
        if not self.validate_step(value):
            raise ValueError(
                f"Step {value} fuori dai limiti [{self.min_closure_step}, {self.max_closure_step}]")

        if self._closure_step != value:
            self._closure_step = value
            self._closure = None  # Forza la rige some se cambia lo step
            self.reset_cache()
    def discretize(self, custom_step: Optional[float] = None):
        if custom_step is not None:
            self.closure_step = custom_step

        self._discretization()
        if not self.closure_is_valid:
            warnings.warn("La discretizzazione non ha prodotto un contorno valido (almeno 3 punti).")

        self.reset_cache()
    @abstractmethod
    def _discretization(self):
        pass

    @staticmethod
    def calc_area(points: np.typing.ArrayLike) -> float:
        points_arr = np.asarray(points, dtype=np.float64)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare l'area: vertici insufficienti")

        x = points_arr[:, 0]
        y = points_arr[:, 1]

        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        double_area = np.sum(x * y_next - x_next * y)
        return 0.5 * double_area
    @staticmethod
    def calc_perimeter(points: np.typing.ArrayLike) -> float:
        points_arr = np.asarray(points, dtype=np.float64)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il perimetro: vertici insufficienti")

        diffs = points_arr - np.roll(points_arr, -1, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return np.sum(dists)
    @staticmethod
    def calc_boundary(points: np.typing.ArrayLike) -> Tuple[float, float, float, float, float, float]:
        points_arr = np.asarray(points)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il boundary: vertici insufficienti")

        x = points_arr[:, 0]
        y = points_arr[:, 1]

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        theta_min, theta_max = 0.0, 0.0

        new_bounding = (x_min, x_max, y_min, y_max, theta_min, theta_max)
        return new_bounding
    @staticmethod
    def calc_barycenter(points: np.typing.ArrayLike) -> np.typing.NDArray[np.float64]:
        points_arr = np.asarray(points)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il baricentro: vertici insufficienti")
        return np.mean(points_arr, axis=0)

    def reset_cache(self):
        self._area = None
        self._perimeter = None
        self._boundary = None
        self._barycenter = None
    def reset(self, resets: Optional[List[Literal["perimeter", "area", "boundary", "barycenter", "all"]]] = None):
        if resets is None:
            self.reset_cache()
            return

        for target in resets:
            if target == "all":
                self.reset_cache()
                return

            elif target == "perimeter":
                self._perimeter = None
            elif target == "area":
                self._area = None
            elif target == "boundary":
                self._boundary = None
            elif target == "barycenter":
                self._barycenter = None

    def translate(self, offset: np.typing.ArrayLike):
        if not self.closure_is_valid:
            raise ValueError("Impossibile traslare: contorno non definito")

        if self.origin is None:
            warnings.warn(f"Warning: riferimento origin non definito")

        self._closure = Shape._translate_points(offset, self.closure)
        if self._barycenter is not None:
            self._barycenter = Shape._translate_points(offset, self._barycenter)

        self.reset(["boundary", "area"])
    def rotate(self, angle_rad: float):
        if not self.closure_is_valid:
            raise ValueError("Impossibile ruotare: contorno non definito")

        ref = self._origin
        if ref is None:
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._closure = Shape._rotate_points(angle_rad, self.closure, ref)
        if self._barycenter is not None:
            self._barycenter = Shape._rotate_points(angle_rad, self._barycenter, ref)

        self.reset(["boundary", "area"])
    def scale(self, factors: np.typing.ArrayLike):
        if not self.closure_is_valid:
            raise ValueError("Impossibile scalare: contorno non definito")

        ref = self._origin
        if ref is None:
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._closure = Shape._scale_points(factors, self.closure, ref)

        self.reset_cache()

    @staticmethod
    def _translate_points(offset: np.typing.ArrayLike, points: np.typing.ArrayLike):
        delta = np.asarray(offset, dtype=np.float64).flatten()
        if delta.shape != (2,):
            raise ValueError("L'offset deve essere un vettore di 2 elementi (dx, dy)")

        return points + delta
    @staticmethod
    def _rotate_points(angle_rad: float, points: np.typing.ArrayLike, ref: np.typing.ArrayLike):
        if np.isclose(angle_rad, 0.0, atol=WorkingSpace.EPS_12):
            return points.copy()

        c, s = np.cos(angle_rad), np.sin(angle_rad)

        centered = points - ref
        x_new = centered[:, 0] * c - centered[:, 1] * s
        y_new = centered[:, 0] * s + centered[:, 1] * c
        centered[:, 0] = x_new
        centered[:, 1] = y_new
        centered += ref
        return centered
    @staticmethod
    def _scale_points(factor, points: np.ndarray, ref: np.ndarray):
        s = np.asarray(factor, dtype=np.float64)
        if s.ndim == 1 and s.size == 2:
            s = s.reshape(1, 2)
        elif s.ndim == 0:
            pass
        elif s.shape != (1, 2):
            raise ValueError(f"Fattore di scala non valido. Atteso scalare o array (2,), ricevuto {s.shape}")

        if np.allclose(s, 1.0, atol=1e-12):
            return points.copy()

        if np.any(np.isclose(s, 0.0, atol=1e-12)):
            warnings.warn("Scaling con fattore zero: la geometria collasserà.")

        return (points - ref) * s + ref

    @staticmethod
    def _get_styled_axis() -> plt.Axes:
        style_params = {
            'font.family': 'serif',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5
        }
        plt.rcParams.update(style_params)

        fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        return ax
    def draw(self, ax: Optional[plt.Axes] = None, show: bool = True, **kwargs) -> Optional[plt.Axes]:
        if self.closure is None or len(self.closure) < 3:
            warnings.warn("Nessun contorno disponibile per il disegno.") # Al posto di print()
            return ax

        if ax is None:
            ax = Shape._get_styled_axis()

        points = np.vstack([self.closure, self.closure[0]])
        x, y = points[:, 0], points[:, 1]

        if not kwargs:
            kwargs = {'color': 'black', 'marker': 'o', 'markersize': 2}

        ax.plot(x, y, **kwargs)
        ax.plot(self.origin[0], self.origin[1], 'rx', label='Origin')
        if show:
            plt.show()

        return ax