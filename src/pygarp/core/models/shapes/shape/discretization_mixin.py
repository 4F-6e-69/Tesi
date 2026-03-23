import warnings
from abc import ABC, abstractmethod

from numpy import typing as npt
import numpy as np
from shapely.geometry import Polygon

from pygarp.core.models.utils import common_divisors_with_tolerance, find_near_divisors, filter_by_tolerance
from pygarp.core.models.commons import Eps, DiscretizationMethod, EpsConfig


class DiscretizationRequirements(ABC):
    @property
    @abstractmethod
    def polygon(self) -> Polygon:
        pass
    @property
    @abstractmethod
    def length(self) -> float:
        pass

    @abstractmethod
    def reset_all(self):
        pass

class DiscretizationMixin(DiscretizationRequirements, ABC):
    shape_order: int = 3

    def __init__(self):
        self._closure: npt.NDArray[np.float64] | None = None
        self._step: float | None = None
        self._step_max: float | None = None
        self._step_min: float | None = None
        self._sure_steps: npt.NDArray[np.float64] | None = None

        super().__init__()

    @property
    def min_step(self) -> float:
        if self._step_min is None:
            self._step_min = self._calc_step_min()
        return self._step_min
    @property
    def max_step(self) -> float:
        if self._step_max is None:
            self._step_max = self._calc_step_max()
        return self._step_max
    @property
    def sure_steps(self) -> npt.NDArray[np.float64] | None:
        if self._sure_steps is None:
            _ = self.max_step
        return self._sure_steps
    @property
    def step(self) -> float | None:
        return self._step

    @step.setter
    def step(self, step: float):
        self.set_step(step)
    def set_step(self, step: float, *, cast: bool = True, eps: EpsConfig | float = Eps.eps10, warn: bool = False):
        step = float(abs(step))
        if step < self.min_step - eps:
            if warn: warnings.warn(f"Passo di discretizzazione troppo piccolo, minimo accettato: {self.min_step}")
            self._step = self.min_step
            return
        if step > self.max_step + eps:
            if warn: warnings.warn(f"Passo di discretizzazione troppo grande, massimo accettato: {self.max_step}")
            self._step = self.max_step
            return

        is_safe = np.any((self.sure_steps > step - eps) & (self.sure_steps < step + eps))
        if not is_safe:
            if not cast:
                if warn: warnings.warn("Il passo impostato non è totalmente sicuro, il percorso calcolato potrebbe subire deformazioni")
            else:
                diffs = np.abs(self.sure_steps - step)
                best_index = np.argmin(diffs)
                step = float(self.sure_steps[best_index])
        self._step = step

    def _calc_step_min(self) -> float:
        order = np.floor(np.log10(self.length) - self.shape_order)
        return 10 ** order
    def _calc_step_max(self) -> float:
        points = np.array(self.polygon.exterior.coords[:-1], dtype=np.float64)
        diffs = points - np.roll(points, 1, axis=0)
        dists = np.round(np.linalg.norm(diffs, axis=1) * (10 ** self.shape_order)).astype(int)

        valid_dists = dists[dists > 0]
        if len(valid_dists) == 0:
            self._sure_steps = np.array([1.0], dtype=np.float64)
            return 1.0

        min_distance_order = np.floor(np.log10(np.min(valid_dists)))
        tolerance = 5 * (10 ** min_distance_order)
        mcd, _ = common_divisors_with_tolerance(dists, tolerance)

        detailed_divisors = find_near_divisors(mcd, 50)
        divisors = filter_by_tolerance(detailed_divisors, 2)
        self._sure_steps = divisors / (10 ** self.shape_order)
        return float(mcd / (10 ** self.shape_order))

    def discretize(self, *, discretization_method: DiscretizationMethod = None, custom_step: float | None = None) -> npt.NDArray[np.float64]:
        if custom_step is not None:
            self.step = custom_step

        if discretization_method is None:
            self._discretization()
        elif discretization_method == "adaptive":
            self._discretization_adaptive()
        elif discretization_method == "uniform":
            self._discretization_uniform()
        else:
            warnings.warn(f"Metodo '{discretization_method}' non valido. Verrà usato il fallback sui vertici.")
            self._discretization()

        return self._closure

    def _discretization(self) -> npt.NDArray[np.float64]:
        x, y = self.polygon.exterior.coords.xy
        self._closure = np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
        return self._closure
    def _discretization_adaptive(self) -> npt.NDArray[np.float64]:
        perimeter = self.polygon.exterior

        if self.step is None:
            _ = self.max_step
            mid_index = len(self.sure_steps) // 2
            self.step = float(self.sure_steps[mid_index])

        step = self.step

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

        if len(points) > 0: points.append(coord[-1])
        self._closure = np.array(points, dtype=np.float64)
        return self._closure
    def _discretization_uniform(self) -> npt.NDArray[np.float64]:
        if self.step is None:
            _ = self.max_step
            mid_index = len(self.sure_steps) // 2
            self.step = float(self.sure_steps[mid_index])

        step = self.step
        coords = np.array(self.polygon.exterior.coords)

        points = [coords[0]]
        current_pt = coords[0]

        seg_idx = 0
        current_t = 0.0

        while seg_idx < len(coords) - 1:
            pa = coords[seg_idx]
            pb = coords[seg_idx + 1]

            pv = pb - pa
            pw = pa - current_pt

            a = np.dot(pv, pv)
            b = 2 * np.dot(pw, pv)
            c = np.dot(pw, pw) - step ** 2

            if a == 0:
                seg_idx += 1
                current_t = 0.0
                continue

            discriminant = b ** 2 - 4 * a * c

            if discriminant >= 0:
                t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                t2 = (-b + np.sqrt(discriminant)) / (2 * a)

                valid_ts = [t for t in (t1, t2) if current_t + 1e-7 < t <= 1.0 + 1e-7]

                if valid_ts:
                    t = min(valid_ts)
                    t_capped = min(t, 1.0)

                    next_pt = pa + t_capped * pv
                    points.append(next_pt)

                    current_pt = next_pt
                    current_t = t_capped
                    continue
            seg_idx += 1
            current_t = 0.0

        self._closure = np.array(points, dtype=np.float64)
        return self._closure