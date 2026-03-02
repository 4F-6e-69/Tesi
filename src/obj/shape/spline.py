from typing import Tuple, Optional
from numpy import typing as nptyping
from scipy.interpolate import splprep, splev, interp1d

from src.obj.shape.shape import Shape
from src.obj.shape.parametric_curve import ParametricCurve
from src.utils import *

from matplotlib import pyplot as plt

class ClosedSpline(ParametricCurve):
    """
    Rappresenta una curva B-Spline chiusa e periodica.

    Implementazione concreta di `ParametricCurve`. Genera un contorno continuo e smussato a partire da un set di punti di controllo.
    Dispone di un sistema a doppio livello di dettaglio: una curva ad alta definizione (HD) per calcoli geometrici precisi, e un contorno discretizzato (closure) campionato a distanze equidistanti per il rendering o altre operazioni discrete.
    """
    def __init__(self, points, smoothness: float = 0.25, high_definition_points: int = 10000):
        super().__init__()
        new_points_array = validate_array_of_2d_coordinates(points)
        if new_points_array is None:
            new_points_array = np.asarray(points)

        spline_data = splprep(new_points_array.T.astype(np.float64), s=smoothness, per=1)
        self.__tck = spline_data[0]
        self.__u = spline_data[1]

        hdu = np.linspace(0, 1, high_definition_points)
        self._high_definition_u = hdu
        hdx, hdy = splev(hdu, self.__tck)

        self._high_definition_closure = np.column_stack((hdx, hdy))

    @property
    def high_definition_u(self):
        """
        :return: Array 1D dei parametri `t` (da 0 a 1) usati per la curva ad alta definizione.
        :rtype: numpy.typing.NDArray[np.float64]
        """
        return self._high_definition_u
    @property
    def high_definition_closure(self):
        """
        :return: Array 2D delle coordinate campionate ad altissima risoluzione.
        :rtype: numpy.typing.NDArray[np.float64]
        """
        return self._high_definition_closure

    @property
    def area_hd(self) -> float:
        """
        Calcola il valore assoluto dell'area della spline ad alta risoluzione.
        Il risultato viene salvato in cache tramite **lazy evaluation**.

        :return: Area strettamente positiva della spline.
        :rtype: float
        """
        if self._area is None:
            self._area = Shape.calc_area(self.high_definition_closure)
        return abs(self._area)
    @property
    def sign_area_hd(self) -> float:
        """
        Calcola l'area algebrica (con segno) della spline ad alta risoluzione.
        Sfrutta la **cache** per ottimizzare le chiamate successive.

        :return: Area della curva (il segno dipende dall'orientamento dei punti).
        :rtype: float
        """
        if self._area is None:
            self._area = Shape.calc_area(self.high_definition_closure)
        return self._area
    @property
    def length(self) -> float:
        """
        Calcola il perimetro del poligono discretizzato (closure standard).

        :return: La lunghezza totale calcolata sui punti del contorno discretizzato.
        :rtype: float
        """
        if self._length is None:
            self._length = Shape.calc_length(self.high_definition_closure)
        return self._length
    @property
    def barycenter_hd(self) -> nptyping.NDArray[np.float64]:
        """
        :return: Array 2D `[x, y]` del baricentro calcolato sui punti ad alta definizione, salvato in **cache**.
        :rtype: nptyping.NDArray[np.float64]
        """
        if self._barycenter is None:
            self._barycenter = Shape.calc_barycenter(self.high_definition_closure)
        return self._barycenter
    @property
    def bounding_box_hd(self) -> Tuple[float, float, float, float, float, float]:
        """
        :return: Tupla con i limiti spaziali della curva ad alta definizione. Salvata in **cache**.
        :rtype: Tuple[float, float, float, float, float, float]
        """
        if self._bounding_box is None:
            self._bounding_box = Shape.calc_bounding_box(self.high_definition_closure)
        return self._bounding_box

    @property
    def t_range(self) -> tuple[float, float]:
        """
        Dominio del parametro della spline.

        :return: Limiti normalizzati `(0.0, 1.0)`.
        :rtype: tuple[float, float]
        """
        return 0., 1.
    def point_at(self, t: ArrayLike) -> nptyping.NDArray[np.float64]:
        r"""
        Valuta le coordinate della spline in corrispondenza del parametro `t`.

        :param t: Vettore o singolo valore del parametro ($t \in [0, 1]$).
        :type t: ArrayLike
        :return: Coordinate `[x, y]` valutate.
        :rtype: numpy.typing.NDArray[np.float64]
        """
        x, y = splev(t if len(t) > 1 else [t], self.__tck)
        return np.column_stack((x, y))

    def discretize(self, custom_step: Optional[float] = None):
        if self.closure_step is None:
            self.closure_step = self.closure_step_max / 10

        super().discretize()
    def _discretization(self):
        """
        Popola la `_closure` generando punti equidistanti lungo la curva (arc-length).

        Utilizza l'interpolazione (`interp1d`) della distanza cumulativa per mappare
        una distribuzione lineare dello spazio sulla parametrizzazione non lineare `u`
        della B-Spline.
        """
        x_fine = self.high_definition_closure[:, 0]
        y_fine = self.high_definition_closure[:, 1]

        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        segment_distances = np.sqrt(dx ** 2 + dy ** 2)
        cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0)

        u_from_dist = interp1d(cumulative_distance, self._high_definition_u)
        num_punti_desiderati = np.ceil(self.length / self.closure_step).astype(int)
        distanza_totale = cumulative_distance[-1]
        distanze_target = np.linspace(0, distanza_totale, num_punti_desiderati)

        u_equidistanti = u_from_dist(distanze_target)
        x_equi, y_equi = splev(u_equidistanti, self.__tck)

        self._closure = np.column_stack((x_equi, y_equi))

    def calc_min_closure_step(self, **kwargs) -> float:
        """
        Calcola dinamicamente il passo minimo ammissibile.
        """
        const_order = 4
        order = int(len(str(self.length).split(".")[0]) - const_order)
        return 10 ** order
    def calc_max_closure_step(self, **kwargs) -> float:
        """
        Calcola il passo massimo limitandolo a un terzo del perimetro totale.
        """
        return self.length / 3 - Eps.eps08

    @Shape.closure_step.setter
    def closure_step(self, closure_step: float) -> None:
        """
        Imposta un nuovo passo per la discretizzazione e resetta il contorno.

        :param closure_step: Nuovo valore per il passo.
        :type closure_step: float
        :raises TypeError: Se il tipo inserito non è compatibile.
        :raises ValueError: Se il passo non supera la validazione geometrica.
        """
        if not isinstance(closure_step, float):
            raise TypeError("Tipo di dato non valido")

        if self.is_valid_step(closure_step):
            if self.closure_step is not None and closure_step > self.closure_step:
                warnings.warn(f"attenzione il passo impostato è maggiore del precedete con conseguente perdita di informazione")

            self._closure_step = closure_step
            self._closure = None
            return

        raise ValueError("Valore non valido")

    def is_valid_step(self, custom_step: float) -> bool:
        """
        Verifica che il passo desiderato rientri nei limiti minimo e massimo.
        """
        if not isinstance(custom_step, (float, int, np.number)):
            raise TypeError(f"Custom step non valido: previsto un numero, ricevuto {type(custom_step).__name__}")

        custom_step = float(custom_step)
        min_limit = self.closure_step_min - Eps.eps08
        max_limit = self.closure_step_max + Eps.eps08

        if custom_step < min_limit or custom_step > max_limit:
            warnings.warn(f"Il passo {custom_step} non rientra nei limiti della forma ({self.closure_step_min}, {self.closure_step_max})")
            return False

        return True

    def translate(self, offset: Tuple[float, float] | ArrayLike = (0, 0)):
        """
        Trasla l'intera spline modificandone i punti di controllo originari.

        Garantisce la massima fedeltà geometrica traslando la definizione analitica della curva e la sua versione ad alta definizione. Invalida la cache al termine.
        """
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
        """
        Ruota l'intera spline attorno all'origine trasformando i punti di controllo.
        """
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
        """
        Scala l'intera spline trasformando analiticamente i punti di controllo.
        """
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
        self._closure_step_max = self.calc_max_closure_step()
        self._closure_step_min = self.calc_min_closure_step()

    def draw(self, ax: plt.Axes = None, show: bool = False, **kwargs):
        """
        Renderizza la spline su un piano cartesiano.

        Disegna due layer:
        1. La curva matematica ad alta definizione (tratteggiata in grigio).
        2. Il contorno discretizzato sovrapposto (seguendo lo stile in `kwargs`).

        :param ax: Asse su cui disegnare. Se **None**, ne viene creato uno.
        :type ax: plt.Axes, opzionale
        :param show: Se **True**, invoca `plt.show()`.
        :type show: bool
        :return: L'oggetto assi utilizzato.
        :rtype: plt.Axes | None
        """
        if self.closure is None or len(self.closure) < 3:
            warnings.warn("Nessun contorno disponibile per il disegno.")  # Al posto di print()
            return ax

        if ax is None:
            ax = Shape.style_graph()

        x, y = self.closure[:, 0].copy(), self.closure[:, 1].copy()
        high_definition_x, high_definition_y = self.high_definition_closure[:, 0].copy(), self.high_definition_closure[
            :, 1].copy()

        ax.plot(high_definition_x, high_definition_y, '--', color='grey')
        ax.plot(x, y, 'o-', **kwargs)

        if show:
            plt.show()

        return ax