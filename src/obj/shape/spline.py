from typing import Tuple
from numpy import typing as nptyping
from scipy.interpolate import splprep, splev, interp1d

from src.obj.shape.shape import Shape
from src.obj.shape.parametric_curve import ParametricCurve
from src.utils import *

from matplotlib import pyplot as plt

class ClosedSpline(ParametricCurve):

    def __init__(self, points, smoothness: float = 0.25, high_definition_points: int = 10000):
        new_points_array = validate_array_of_2d_coordinates(points)
        points_x = new_points_array[:, 0]
        points_y = new_points_array[:, 1]

        spline = splprep(np.vstack((points_x, points_y), dtype=np.float64), s=smoothness, per=1)
        self.__tck = spline[0]
        self.__u = spline[1]

        hdu = np.linspace(0, 1, high_definition_points)
        self._high_definition_u = hdu
        hdx, hdy = splev(hdu, self.__tck)
        self._high_definition_closure = np.column_stack((hdx, hdy))
        super().__init__()

    @property
    def high_definition_u(self):
        return self._high_definition_u
    @property
    def high_definition_closure(self):
        return self._high_definition_closure

    @property
    def area_hd(self) -> float:
        if self._area is None:
            self._ensure_property("area ad alta definizione")
            self._area = Shape.calc_area(self.high_definition_closure)
        return abs(self._area)
    @property
    def sign_area_hd(self) -> float:
        if self._area is None:
            self._ensure_property("area con segno ad alta definizione")
            self._area = Shape.calc_area(self.high_definition_closure)
        return self._area
    @property
    def length(self) -> float:
        if self._length is None:
            self._ensure_property("lunghezza")
            self._length = Shape.calc_length(self.closure)
        return self._length
    @property
    def barycenter_hd(self) -> nptyping.NDArray[np.float64]:
        if self._barycenter is None:
            self._ensure_property("baricentro ad alta definizione")
            self._barycenter = Shape.calc_barycenter(self.high_definition_closure)
        return self._barycenter
    @property
    def bounding_box_hd(self) -> Tuple[float, float, float, float, float, float]:
        if self._bounding_box is None:
            self._ensure_property("bounding box ad alta definizione")
            self._bounding_box = Shape.calc_bounding_box(self.high_definition_closure)
        return self._bounding_box

    @property
    def t_range(self) -> tuple[float, float]:
        return 0., 1.
    def point_at(self, t: ArrayLike) -> nptyping.NDArray[np.float64]:
        x, y = splev(t, self.__tck)
        return np.column_stack((x, y))

    def _discretization(self):
        x_fine = self.high_definition_closure[:, 0]
        y_fine = self.high_definition_closure[:, 1]

        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        segment_distances = np.sqrt(dx ** 2 + dy ** 2)
        cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0)

        u_from_dist = interp1d(cumulative_distance, self._high_definition_u)
        num_punti_desiderati = 80
        distanza_totale = cumulative_distance[-1]
        distanze_target = np.linspace(0, distanza_totale, num_punti_desiderati)

        u_equidistanti = u_from_dist(distanze_target)
        x_equi, y_equi = splev(u_equidistanti, self.__tck)

        self._closure = np.column_stack((x_equi, y_equi))

    def calc_min_closure_step(self, points: ArrayLike) -> float:
        const_order = 4
        order = int(len(str(self.length).split(".")[0]) - const_order)
        return 10 ** order
    def calc_max_closure_step(self, points: ArrayLike) -> float:
        return self.length / 3

    @Shape.closure_step.setter
    def closure_step(self, closure_step: float) -> None:
        if not isinstance(closure_step, float):
            raise TypeError("Tipo di dato non valido")

        if self.is_valid_step(closure_step):
            if closure_step > self.closure_step:
                warnings.warn(f"attenzione il passo impostato è maggiore del precedete con conseguente perdita di informazione")

            self._closure_step = closure_step
            self._closure = None
            return

        raise ValueError("Valore non valido")

    def is_valid_step(self, custom_step: float) -> bool:
        if not isinstance(custom_step, (float, int, np.number)):
            raise TypeError(f"Custom step non valido: previsto un numero, ricevuto {type(custom_step).__name__}")

        custom_step = float(custom_step)
        min_limit = self.closure_step_min - Eps().eps08
        max_limit = self.closure_step_max + Eps().eps08

        if custom_step < min_limit or custom_step > max_limit:
            warnings.warn(f"Il passo {custom_step} non rientra nei limiti della forma ({self.closure_step_min}, {self.closure_step_max})")
            return False

        return True

    def translate(self, offset: Tuple[float, float] | ArrayLike = (0, 0)):
        t, c, k = self.__tck
        control_points_2d = np.column_stack((c[0], c[1]))
        new_control_points = Shape.translate_points(control_points_2d, offset)
        new_c = [new_control_points[:, 0], new_control_points[:, 1]]
        self.__tck = (t, new_c, k)

        self._high_definition_closure = Shape.translate_points(self.high_definition_closure, offset)

        try:
            super().translate(offset)
        except ValueError:
            warnings.warn(f"Attenzione! traslazione completata con successo sulla closure ad alta definizione tuttavia il parametro resta none")

        self.reset_cache()
    def rotate(self, angle: float = 0, is_radiant: bool = True, **kwargs):
        ref = self._check_origin(False, np.zeros(2).flatten())

        t, c, k = self.__tck
        control_points_2d = np.column_stack((c[0], c[1]))
        new_control_points = Shape.rotate_points(control_points_2d, ref, angle, is_radiant)
        new_c = [new_control_points[:, 0], new_control_points[:, 1]]
        self.__tck = (t, new_c, k)

        self._high_definition_closure = Shape.rotate_points(self.high_definition_closure, ref, angle, is_radiant)

        try:
            super().rotate(angle, is_radiant, True, ref)
        except ValueError:
            warnings.warn(f"Attenzione! rotazione completata con successo sulla closure ad alta definizione tuttavia il parametro resta none")

        self.reset_cache()
    def scale(self, factor: Tuple[float, float] | ArrayLike | float = 1, **kwargs):
        ref = self._check_origin(False, np.zeros(2).flatten())

        t, c, k = self.__tck
        control_points_2d = np.column_stack((c[0], c[1]))
        new_control_points = Shape.scale_points(control_points_2d, ref, factor)
        new_c = [new_control_points[:, 0], new_control_points[:, 1]]
        self.__tck = (t, new_c, k)

        self._high_definition_closure = Shape.scale_points(self.high_definition_closure, ref, factor)

        try:
            super().scale(factor, True, ref)
        except ValueError:
            warnings.warn(
                f"Attenzione! scalata completata con successo sulla closure ad alta definizione tuttavia il parametro resta none")

        self.reset_cache()

    def draw(self, ax: plt.Axes = None, show: bool = False, **kwargs):
        if self.closure is None or len(self.closure) < 3:
            warnings.warn("Nessun contorno disponibile per il disegno.")  # Al posto di print()
            return ax

        if ax is None:
            ax = Shape._get_styled_axis()

        x, y = self.closure[:, 0].copy(), self.closure[:, 1].copy()
        high_definition_x, high_definition_y = self.high_definition_closure[:, 0].copy(), self.high_definition_closure[
            :, 1].copy()

        ax.plot(high_definition_x, high_definition_y, '--', color='grey')
        ax.plot(x, y, **kwargs)

        if show:
            plt.show()

        return ax