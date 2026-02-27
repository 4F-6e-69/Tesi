from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from numpy import typing as nptyping

from typing import Tuple, List, Literal, Optional
from src.utils import *

Resets = Union[List[Literal["all", "length", "area", "bounding_box", "barycenter"]], Literal["all", "length", "area", "bounding_box", "barycenter"], None]

class Shape(ABC):
    def __init__(self):
        self._origin: nptyping.NDArray[np.float64] = np.zeros(2).flatten()
        self._origin_is_center: bool = False

        self._barycenter: nptyping.NDArray[np.float64] | None = None
        self._length: float | None = None
        self._area: float | None = None
        self._bounding_box: Tuple[float, float, float, float, float, float] | None = None

        self._closure: nptyping.NDArray[np.float64] | None = None
        self._closure_step: float | None = None
        self._closure_step_min: float | None = None
        self._closure_step_max: float | None = None

    @property
    def origin(self) -> nptyping.NDArray[np.float64]:
        return self._origin
    @property
    def origin_is_center(self) -> bool:
        return self._origin_is_center
    @origin.setter
    def origin(self, origin: ArrayLike):
        no = np.asarray(origin, dtype=np.float64, copy=True).flatten()
        if no.shape != self._origin.shape:
            raise ValueError(f"Dimensione errata: attesa {self._origin.shape}, ricevuta {no.shape}")

        self._set_origin_is_center()
    def _set_origin_is_center(self):
        try:
            if self._is_valid_closure():
                self._origin_is_center = np.allclose(self.origin, self.barycenter, atol=Eps.eps08)
            else:
                self._origin_is_center = False
        except ValueError:
            self._origin_is_center = False

    @property
    def area(self) -> float:
        if self._area is None:
            self._ensure_property("area")
            self._area = Shape.calc_area(self.closure)
        return self._area
    @property
    def sign_area(self) -> float:
        if self._area is None:
            self._ensure_property("area con segno")
            self._area = Shape.calc_area(self.closure)
        return abs(self._area)
    @property
    def length(self) -> float:
        if self._length is None:
            self._ensure_property("lunghezza")
            self._length = Shape.calc_length(self.closure)
        return self._length
    @property
    def barycenter(self) -> nptyping.NDArray[np.float64]:
        if self._barycenter is None:
            self._ensure_property("baricentro")
            self._barycenter = Shape.calc_barycenter(self.closure)
        return self._barycenter
    @property
    def bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        if self._bounding_box is None:
            self._ensure_property("bounding box")
            self._bounding_box = Shape.calc_bounding_box(self.closure)
        return self._bounding_box

    @staticmethod
    def calc_length(points: ArrayLike) -> float:
        points_arr = np.asarray(points, dtype=np.float64)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il perimetro: vertici insufficienti")

        diffs = points_arr - np.roll(points_arr, -1, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return np.sum(dists)
    @staticmethod
    def calc_area(points: ArrayLike) -> float:
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
    def calc_bounding_box(points: ArrayLike) -> Tuple[float, float, float, float, float, float]:
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
    def calc_barycenter(points: ArrayLike) -> nptyping.NDArray[np.float64]:
        points_arr = np.asarray(points)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il baricentro: vertici insufficienti")
        return np.mean(points_arr, axis=0)

    @property
    def closure(self) -> nptyping.NDArray[np.float64] | None:
        return self._closure
    @property
    def closure_step(self) -> float | None:
        return self._closure_step
    @property
    def closure_step_min(self) -> float:
        if self._closure_step_min is None:
            pass
        return self._closure_step_min
    @property
    def closure_step_max(self) -> float:
        if self._closure_step_max is None:
            pass
        return self._closure_step_max

    def discretize(self, custom_step: Optional[float] = None):
        if np.isclose(custom_step, self._closure_step, atol=Eps.eps08):
            try:
                self.closure_step = custom_step
            except TypeError or ValueError:
                pass

        self._discretization()
        if not self._is_valid_closure():
            warnings.warn("La discretizzazione non ha prodotto un contorno valido (almeno 3 punti).")

        self.reset_cache()
    @abstractmethod
    def _discretization(self):
        pass

    @abstractmethod
    def calc_min_closure_step(self, points: ArrayLike) -> nptyping.NDArray[np.float64]:
        pass
    @abstractmethod
    def calc_max_closure_step(self, points: ArrayLike) -> nptyping.NDArray[np.float64]:
        pass
    @abstractmethod
    @closure_step.setter
    def closure_step(self, closure_step: float) -> None:
        pass

    def _ensure_property(self, value: str):
        if not self._is_valid_closure():
            n_points = len(self._closure) if self._closure is not None else 0
            raise ValueError(f"Impossibile calcolare {value}: servono almeno 3 punti (trovati: {n_points}).")
    def _is_valid_closure(self) -> bool:
        try :
            result = validate_array_of_2d_coordinates(self.closure)
            if result is not None:
                self._closure = result
            return True
        except ValueError | TypeError :
            return False

    @abstractmethod
    def is_valid_step(self, custom_step: float) -> bool:
        pass

    def translate(self, offset: Tuple[float, float] | ArrayLike = (0, 0)):
        if not self._is_valid_closure():
            raise ValueError("Impossibile traslare: contorno non definito")

        '''
        result = validate_2d_coordinates(self.origin)
        if result is not None:
            self.origin = result
            '''

        self._closure = Shape.translate_points(self.closure, offset)
        self.reset(["bounding_box", "area", "barycenter"])
    def rotate(self, angle: float = 0, is_radiant: bool = True):
        if not self._is_valid_closure():
            raise ValueError("Impossibile traslare: contorno non definito")

        try:
            result = validate_2d_coordinates(self.origin)
            if result is not None:
                self.origin = result

            ref = self.origin
        except TypeError :
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._closure = Shape.rotate_points(self.closure, ref, angle, is_radiant)
        self.reset(["bounding_box", "area", "barycenter"])
    def scale(self, factor: Tuple[float, float] | ArrayLike | float = 1):
        if not self._is_valid_closure():
            raise ValueError("Impossibile scalare: contorno non definito")

        try:
            result = validate_2d_coordinates(self.origin)
            if result is not None:
                self.origin = result

            ref = self.origin
        except TypeError:
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._closure = Shape.scale_points(self.closure, ref, factor)
        self.reset_cache()

    @staticmethod
    def translate_points(points: ArrayLike, offset: Tuple[float, float] | ArrayLike = (0, 0)) -> nptyping.NDArray[np.float64]:
        delta = np.asarray(offset, dtype=np.float64).flatten()
        coords = validate_2d_coordinates(points)

        if delta.shape != (2,):
            raise ValueError("L'offset deve essere un vettore di 2 elementi (dx, dy)")

        return coords + delta
    @staticmethod
    def rotate_points(points: ArrayLike, ref: ArrayLike, angle: float = 0, is_radiant: bool = True) -> nptyping.NDArray[np.float64]:
        angle_rad = angle if is_radiant else np.deg2rad(angle)
        coords = validate_2d_coordinates(points)

        if np.isclose(angle_rad, 0.0, atol=Eps.eps08):
            return coords.copy()

        c, s = np.cos(angle_rad), np.sin(angle_rad)

        centered = coords - ref
        x_new = centered[:, 0] * c - centered[:, 1] * s
        y_new = centered[:, 0] * s + centered[:, 1] * c
        centered[:, 0] = x_new
        centered[:, 1] = y_new
        centered += ref
        return centered
    @staticmethod
    def scale_points(points: ArrayLike, ref: ArrayLike, factor: Tuple[float, float] | ArrayLike | float = 1) -> nptyping.NDArray[np.float64]:
        s = np.asarray(factor, dtype=np.float64)
        coords = validate_2d_coordinates(points)

        if s.ndim == 1 and s.size == 2:
            s = s.reshape(1, 2)
        elif s.ndim == 0:
            pass
        elif s.shape != (1, 2):
            raise ValueError(f"Fattore di scala non valido. Atteso scalare o array (2,), ricevuto {s.shape}")

        if np.allclose(s, 1.0, atol=1e-12):
            return coords.copy()

        if np.any(np.isclose(s, 0.0, atol=1e-12)):
            warnings.warn("Scaling con fattore zero: la geometria collasserà.")

        return (coords - ref) * s + ref

    def reset_cache(self):
        self._area = None
        self._length = None
        self._bounding_box = None
        self._barycenter = None
    def reset(self, resets: Resets):
        if resets is None:
            self.reset_cache()
            return

        for target in resets:
            if target == "all":
                self.reset_cache()
                return

            elif target == "length":
                self._length = None
            elif target == "area":
                self._area = None
            elif target == "boundary":
                self._bounding_box = None
            elif target == "barycenter":
                self._barycenter = None

    @staticmethod
    def style_graph() -> plt.Axes:
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
    def draw(self, ax: plt.Axes, points: ArrayLike, show: bool = False, **kwargs) -> plt.Axes | None:

        ax = Shape.style_graph() if ax is None else ax
        if self._is_valid_closure():
            warnings.warn("Nessun contorno disponibile per il disegno.")  # Al posto di print()
            return ax

        try:
            if points is None:
                other_coords = None
            else:
                other_coords = validate_2d_coordinates(points)
        except ValueError or TypeError:
            other_coords = None
        if other_coords is not None:
            x_other = other_coords[:, 0]
            y_other = other_coords[:, 1]

            ax.scatter(x_other, y_other, 'o', "green")


        points = np.vstack([self.closure, self.closure[0]])
        x, y = points[:, 0], points[:, 1]

        if not kwargs:
            kwargs = {'color': 'black', 'marker': 'o', 'markersize': 2}

        ax.plot(x, y, **kwargs)
        ax.plot(self.origin[0], self.origin[1], 'rx', label='Origin')
        if show:
            plt.show()

        return ax