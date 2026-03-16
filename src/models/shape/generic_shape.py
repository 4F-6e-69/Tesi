import warnings

import numpy as np
from numpy import typing as npt

from shapely.geometry import Polygon, Point
from shapely import affinity

from src.utils import Eps
from src.utils import ArrayLike, Resets, Ref, DiscretizationMethod
from src.utils import validate_array_of_2d_coordinates, validate_2d_coordinates
from src.utils import _tolerated_mcd, filter_arrays_tolerance, _all_almost_divisors

class Shape:
    shape_order: int = 4

    def __init__(self, points: ArrayLike, origin: ArrayLike = None, **kwargs) -> None:
        __skip = kwargs.get("__skip", False)
        __order = kwargs.get("__order", False)

        # validazione
        if not __skip:
            validated_points = validate_array_of_2d_coordinates(points)
            points = points if validated_points is None else validated_points

            if origin is not None:
                validated_origin = validate_2d_coordinates(origin)
                origin = origin if validated_origin is None else validated_origin

        # Ordinamento dei punti per evitare concavità / auto-intersezioni strane
        if __order:
            # TODO implementazione di un algoritmo per ordinare i punti secondo un certo rodine CW o CCW
            pass

        # Creazione della forma con Polygon per calcoli geometri e planari
        shape: Polygon = Polygon(points)
        self._shapely_shape = shape

        # Inizializzazione del riferimento
        centroid_pt = shape.centroid
        center_coord = np.array([centroid_pt.x, centroid_pt.y], dtype=np.float64)
        if origin is None:
            self._origin = np.zeros(2, dtype=np.float64)
        else:
            self._origin = np.asarray(origin, dtype=np.float64)
        self._origin_is_center: bool = np.allclose(self._origin, center_coord)

        # Inizializzazione dell cache a None
        self._area: float | None = None
        self._length: float | None = None
        self._bounds: tuple[float, float, float, float] | None = None
        self._barycenter: npt.NDArray[np.float64] | None = None

        # Inizializzazione della cache relativo al contorno della forma
        self._closure: npt.NDArray[np.float64] | None = None
        self._discretization_step: float | None = None
        self._sure_steps: npt.NDArray[np.float64] | None = None
        self._max_discretization_step: float | None = None
        self._min_discretization_step: float | None = None

    @property
    def shapely(self) -> Polygon | None:
        if self._shapely_shape is None:
            warnings.warn("L'oggetto Shapely non è stato ancora inizializzato (output None)")
            return None
        return self._shapely_shape

    @property
    def origin(self) -> npt.NDArray[np.float64] | None:
        return self._origin
    @origin.setter
    def origin(self, origin: ArrayLike, **kwargs) -> None:
        __skip = kwargs.get("__skip", False)

        if not __skip:
            try:
                validated_origin = validate_2d_coordinates(origin)
                origin = origin if validated_origin is None else validated_origin
            except (TypeError, ValueError):
                warnings.warn(
                    f"Origine fornita: {origin} non valida - Il precedente riferimento non è stato modificato")
                return

        self._origin = np.asarray(origin, dtype=np.float64)
        self._origin_is_center = np.allclose(self.barycenter, self._origin)
    @property
    def origin_is_center(self) -> bool | None:
        return self._origin_is_center

    @property
    def ccw(self):
        return self.shapely.exterior.is_ccw
    @property
    def area(self) -> float | None:
        if self._area is None:
            self._area = self.shapely.area
        return self._area
    @property
    def length(self) -> float | None:
        if self._length is None:
            self._length = self.shapely.exterior.length
        return self._length
    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        if self._bounds is None:
            self._bounds = self.shapely.bounds
        return self._bounds
    @property
    def barycenter(self) -> npt.NDArray[np.float64] | None:
        if self._barycenter is None:
            self._barycenter = np.asarray([self.shapely.centroid.x, self.shapely.centroid.y], dtype=np.float64)
        return self._barycenter

    @property
    def closed_control_points(self):
        x, y = self.shapely.exterior.coords.xy
        return np.column_stack((x, y))
    @property
    def control_points(self):
        x, y = self.shapely.exterior.coords.xy
        return np.delete(np.column_stack((x, y)), -1)

    @property
    def min_discretization_step(self) -> float:
        if self._min_discretization_step is None:
            self._min_discretization_step = self._calc_min_discretization_step()

        if isinstance(self, Shape):
            warnings.warn("Il robot è in grado di gestire da se la velocità di conseguenza verranno passati i vertici come punto di controllo")

        return self._min_discretization_step
    @property
    def max_discretization_step(self) -> float:
        if self._max_discretization_step is None:
            self._max_discretization_step = self._calc_max_discretization_step()

        if isinstance(self, Shape):
            warnings.warn(
                "Il robot è in grado di gestire da se la velocità di conseguenza verranno passati i vertici come punto di controllo")

        return self._max_discretization_step
    @property
    def sure_steps(self) -> npt.NDArray[np.float64] | None:
        if self._sure_steps is None:
            warnings.warn("I passi di discretizzazione sicura della figura non sono ancora stati definiti")

        if isinstance(self, Shape):
            warnings.warn(
                "Il robot è in grado di gestire da se la velocità di conseguenza verranno passati i vertici come punto di controllo")

        return self._sure_steps
    @property
    def discretization_step(self) -> float | None:
        if self._discretization_step is None:
            warnings.warn("Passo di discretizzazione della figura non ancora definito")

        if isinstance(self, Shape):
            warnings.warn(
                "Il robot è in grado di gestire da se la velocità di conseguenza verranno passati i vertici come punto di controllo")

        return self._discretization_step
    @discretization_step.setter
    def discretization_step(self, step: float, **kwargs) -> None:
        __cast = kwargs.get("__cast", True)
        __epsilon = kwargs.get("__epsilon", Eps.eps12)

        step = float(abs(step))
        if step < self.min_discretization_step - __epsilon:
            warnings.warn(f"Passo di discretizzazione troppo piccolo, minimo accettato: {self.min_discretization_step}")
            self._discretization_step = self.min_discretization_step
            return
        if step > self.max_discretization_step + __epsilon:
            warnings.warn(f"Passo di discretizzazione troppo grande, massimo accettato: {self.max_discretization_step}")
            self._discretization_step = self.max_discretization_step
            return

        is_safe = np.any((self.sure_steps > step - __epsilon) & (self.sure_steps < step + __epsilon))
        if not is_safe:
            if not __cast:
                warnings.warn("Il passo impostato non è totalmente sicuro, il percorso calcolato potrebbe subire deformazioni")
            else:
                diffs = np.abs(self.sure_steps - step)
                best_index = np.argmin(diffs)
                step = float(self.sure_steps[best_index])

        self._discretization_step = step

        if isinstance(self, Shape):
            warnings.warn(
                "Il robot è in grado di gestire da se la velocità di conseguenza verranno passati i vertici come punto di controllo...  il passo di discretizzazione calcolato non verrà usato")

    def _calc_min_discretization_step(self, **kwargs) -> float:
        __len: float | None = kwargs.get("__len", None)
        order = int(np.log10(self.length) - Shape.shape_order)
        return 10 ** order
    def _calc_max_discretization_step(self, **kwargs) -> float:
        x, y = self.shapely.exterior.coords.xy
        points = np.delete(np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))), -1, axis=0)
        diffs = points - np.roll(points, 1, axis=0)
        dists = np.round(np.linalg.norm(diffs, axis=1) * (10 ** Shape.shape_order)).astype(int)

        min_distance_order = np.log10(np.min(dists))

        mcd, _ = _tolerated_mcd(dists, 5 * 10 ** min_distance_order)
        detailed_divisors = _all_almost_divisors(mcd, 50)
        divisors = filter_arrays_tolerance(detailed_divisors, 2)

        self._sure_steps = divisors / 10 ** Shape.shape_order
        return mcd / 10 ** Shape.shape_order

    def discretize(self, **kwargs) -> npt.NDArray[np.float64]:
        __custom_step = kwargs.get("__custom_step", None)
        __discretization_method: DiscretizationMethod = kwargs.get("__discretization_method", None)

        if __discretization_method is None:
            self._discretization()
            self.reset_cache()
            return self._closure

        if __custom_step is not None:
            self.discretization_step = __custom_step

        if __discretization_method is "adaptive":
            self._discretization_adaptive()
        elif __discretization_method is "uniform":
            self._discretization_uniform()
        else:
            self._discretization()
            warnings.warn(f"{__discretization_method} non valido, metodo di discretizzazione restituito NONE")

        self.reset_cache()

        return self._closure
    def _discretization(self):
        x, y = self.shapely.exterior.coords.xy
        return np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    def _discretization_adaptive(self):
        perimeter = self.shapely.exterior

        if self.discretization_step is None:
            _ = self.max_discretization_step
            mid_index = len(self.sure_steps) // 2
            self.discretization_step = float(self.sure_steps[mid_index])

        step = self.discretization_step

        coord = np.array(perimeter.coords)
        points = []

        for i in range(len(coord) - 1):
            p1 = coord[i]
            p2 = coord[i + 1]

            segment_vector = p2 - p1
            segment_length = np.linalg.norm(segment_vector)

            num_steps = int(np.round(segment_length / step))

            if num_steps == 0:
                continue

            for j in range(num_steps):
                fraction = j / num_steps
                new_point = p1 + (segment_vector * fraction)
                points.append(new_point)

        self._closure = np.array(points, dtype=np.float64)
        return self._closure
    def _discretization_uniform(self) -> npt.NDArray[np.float64]:
        if self.discretization_step is None:
            _ = self.max_discretization_step
            mid_index = len(self.sure_steps) // 2
            self.discretization_step = float(self.sure_steps[mid_index])

        step = self.discretization_step
        coords = np.array(self.shapely.exterior.coords)

        points = [coords[0]]
        current_pt = coords[0]

        seg_idx = 0
        current_t = 0.0

        while seg_idx < len(coords) - 1:
            A = coords[seg_idx]
            B = coords[seg_idx + 1]

            V = B - A
            W = A - current_pt

            a = np.dot(V, V)
            b = 2 * np.dot(W, V)
            c = np.dot(W, W) - step ** 2

            if a == 0:
                seg_idx += 1
                current_t = 0.0
                continue

            discriminant = b ** 2 - 4 * a * c

            if discriminant >= 0:
                t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                t2 = (-b + np.sqrt(discriminant)) / (2 * a)

                valid_ts = [t for t in (t1, t2) if 1.0 + 1e-7 < t > current_t + 1e-7]
                if valid_ts:
                    t = min(valid_ts)
                    t_capped = min(t, 1.0)

                    next_pt = A + t_capped * V
                    points.append(next_pt)

                    current_pt = next_pt
                    current_t = t_capped
                    continue

            seg_idx += 1
            current_t = 0.0

        self._closure = np.array(points, dtype=np.float64)
        return self._closure

    def reset_cache(self):
        self._area = None
        self._length = None
        self._bounds = None
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
                self._min_discretization_step = None
                self._max_discretization_step = None
                self._discretization_step = None
            elif target == "area":
                self._area = None
            elif target == "bounds":
                self._bounds = None
            elif target == "barycenter":
                self._barycenter = None

            elif target == "closure":
                self._closure = None
            elif target == "step":
                self._discretization_step = None

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> 'Shape':
        self._shapely_shape = affinity.translate(self.shapely, xoff=x_off, yoff=y_off)

        if self._origin is not None:
            self._origin = self._origin + np.array([x_off, y_off])

        self.reset(["bounds", "barycenter", "closure"])
        return self
    def rotate(self, angle: float = 0.0, ref: Ref = "origin") -> 'Shape':
        ref_origin = self.origin if ref == "origin" else self.barycenter
        self._shapely_shape = affinity.rotate(self.shapely, angle=angle, origin=(ref_origin[0], ref_origin[1]))

        if ref == "barycenter":
            orig_point = affinity.rotate(Point(self._origin), angle=angle, origin=(ref_origin[0], ref_origin[1]))
            self._origin = np.array([orig_point.x, orig_point.y], dtype=np.float64)

        self.reset(["bounds", "barycenter", "closure"])
        return self
    def scale(self, x_fact: float = 1.0, y_fact: float = 1.0, ref: Ref = "origin") -> 'Shape':
        ref_origin = self.origin if ref == "origin" else self.barycenter

        self._shapely_shape = affinity.scale(
            self.shapely,
            xfact=x_fact,
            yfact=y_fact,
            origin=(ref_origin[0], ref_origin[1])
        )

        if ref == "barycenter" and self._origin is not None:
            orig_point = affinity.scale(
                Point(self._origin),
                xfact=x_fact,
                yfact=y_fact,
                origin=(ref_origin[0], ref_origin[1])
            )
            self._origin = np.array([orig_point.x, orig_point.y], dtype=np.float64)

        self.reset_cache()
        self.reset(["length", "closure"])
        return self

    def __str__(self):
        return self._shapely_shape.__str__().split(" ")[0]
    def __repr__(self):
        return self._shapely_shape.__repr__()