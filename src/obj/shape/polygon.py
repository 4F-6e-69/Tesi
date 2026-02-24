import numpy as np
import math
import warnings

from src.obj.plane.working_plane import WorkingSpace
from src.obj.shape.shape import Shape

from matplotlib import patches
from matplotlib import pyplot as plt


class Polygon(Shape):
    def __init__(self, vertices: np.typing.ArrayLike[np.float64], origin: np.typing.ArrayLike[np.float64] = np.asarray([0, 0], dtype=float)):
        super().__init__()

        self._vertices: np.typing.NDArray[np.float64] = np.asarray(vertices, dtype=np.float64)
        self.origin = origin

        precision = int(math.ceil(-np.log10(self.tolerance)))
        self._max_closure_step = Polygon._calc_max_closure_step(self._vertices, precision)
        self._min_closure_step = Polygon._calc_min_closure_step(self.max_closure_step, self.tolerance)

    @classmethod
    def new_polygon(cls, vertices: np.ndarray, origin_is_center: bool):
        np_vertices = np.asarray(vertices, dtype=np.float64)
        origin = np.mean(np_vertices, axis=0) if origin_is_center else np.zeros(2)

        return cls(np_vertices, origin)

    def _set_origin_is_center(self):
        try:
            if self.vertices_are_valid:
                center = self.barycenter
                self._origin_is_center = np.allclose(self.origin, center, atol=self.tolerance)
            else:
                self._origin_is_center = False
        except ValueError:
            self._origin_is_center = False

    @property
    def perimeter(self) -> float | None:
        if self._perimeter is None:
            self._ensure_valid_vertices("perimetro")
            self._perimeter = Shape.calc_perimeter(self.vertices)
        return self._perimeter
    @property
    def area(self) -> float:
        if self._area is None:
            self._ensure_valid_vertices("area")
            self._area = Shape.calc_area(self.vertices)
        return self._area
    @property
    def boundary(self) -> tuple[float, float, float, float, float, float] | None:
        if self._boundary is None:
            self._ensure_valid_vertices("boundary")
            self._boundary = Shape.calc_boundary(self.vertices)
        return self._boundary
    @property
    def barycenter(self) -> np.typing.NDArray[np.float64]:
        if self._barycenter is None:
            self._ensure_valid_vertices("baricentro")
            self._barycenter = Shape.calc_barycenter(self.vertices)
        return self._barycenter

    @property
    def vertices(self) -> np.typing.NDArray[np.float64] | None:
        return self._vertices
    @property
    def vertices_are_valid(self) -> bool:
        return (self._vertices is not None and
                isinstance(self._vertices, np.ndarray) and
                self._vertices.shape[0] >= 3)
    def _ensure_valid_vertices(self, property_name: str):
        if not self.vertices_are_valid:
            n_points = len(self._vertices) if self._vertices is not None else 0
            raise ValueError(f"Impossibile calcolare {property_name}: servono almeno 3 punti (trovati: {n_points}).")

    @staticmethod
    def _calc_max_closure_step(vertices: np.typing.NDArray[np.float64], precision: int = 6) -> float | None:
        diffs = np.asarray(vertices, np.float64) - np.roll(np.asarray(vertices, np.float64), -1, axis=0)
        multiplier = 10 ** precision

        lengths_int = np.round(diffs * multiplier).astype(np.int64)
        flat_lengths = lengths_int.flatten()

        flat_lengths = np.abs(flat_lengths)
        gcd_int = np.gcd.reduce(flat_lengths)

        return float(gcd_int.item()) / multiplier
    @staticmethod
    def _calc_min_closure_step(max_step: float, precision: int | float = 6) -> float | None:
        if max_step is None:
            raise ValueError("Impossibile calcolare il minimo step: max_closure_step mancante")

        tolerance = float()

        if isinstance(precision, int):
            tolerance = float(10 ** -precision)
        elif isinstance(precision, float):
            if precision < WorkingSpace.EPS_05:
                tolerance = precision
            else:
                warnings.warn("Tolleranza troppo elevata, calcoli inconsistenti")
        else:
            raise TypeError("Precisione non valida")

        temp_minimus = max_step / 10
        if temp_minimus < tolerance:
            return tolerance

        return temp_minimus

    def _discretization(self):
        if self.closure_step is None:
            self._closure = self._vertices.copy()
            return

        points = self._vertices
        num_corners = len(points)

        diffs = points - np.roll(points, -1, axis=0)
        real_lengths = np.linalg.norm(diffs, axis=1)

        ratios = real_lengths / self.closure_step

        num_segments = np.round(ratios).astype(np.int64)
        residuals = np.abs(ratios - num_segments)

        if np.any(residuals > self.tolerance):
            failed_idx = np.argmax(residuals)
            raise ValueError(
                f"Lo step {self.closure_step} non è compatibile con la geometria."
                f"\nLato {failed_idx} (L={real_lengths[failed_idx]:.4f}) non divisibile per step."
                f"\nResiduo: {residuals[failed_idx]:.2e} > Tolleranza {self.tolerance}"
            )

        new_points_list = []

        for i in range(num_corners):
            p_start = points[i]
            p_end = points[(i + 1) % num_corners]  # Gestione wraparound ultimo punto
            n_seg = num_segments[i]

            if n_seg < 1:
                continue

            points_on_side = np.linspace(p_start, p_end, int(n_seg), endpoint=False)
            new_points_list.append(points_on_side)

        if len(new_points_list) > 0:
            self._closure = np.vstack(new_points_list)
        else:
            self._closure = self._vertices.copy()

    def translate(self, offset: np.typing.ArrayLike[np.float64]):
        self._ensure_valid_vertices("traslazione")

        if self.origin is None:
            warnings.warn(f"Warning: riferimento origin non definito")

        self._vertices = Shape._translate_points(offset, self.vertices)
        if self._barycenter is not None:
            self._barycenter = Shape._translate_points(offset, self._barycenter)

        self.reset(["boundary"])
        self._closure = None
    def rotate(self, angle_rad: float):
        self._ensure_valid_vertices("rotazione")

        ref = self._origin
        if ref is None:
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._vertices = Shape._rotate_points(angle_rad, self.vertices, ref)
        if self._barycenter is not None:
            self._barycenter = Shape._rotate_points(angle_rad, self._barycenter, ref)

        self.reset(["boundary"])
        self._closure = None
    def scale(self, factors: np.typing.ArrayLike[np.float64]):
        self._ensure_valid_vertices("scala")

        ref = self._origin
        if ref is None:
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._vertices = Shape._scale_points(factors, self.vertices, ref)

        self.reset_cache()
        self._closure = None

    def draw(self, ax: plt.Axes | None = None, show: bool = True, **kwargs) -> plt.Axes | None:
        if ax is None:
            ax = Shape._get_styled_axis()

        xs, ys = zip(*self.vertices)

        if not kwargs:
            kwargs = {'facecolor': '#2c3e50', 'alpha': 0.3, 'edgecolor': '#2c3e50', 'zorder': 2}

        polygon = patches.Polygon(self.vertices, **kwargs)
        ax.add_patch(polygon)

        ax.scatter(xs, ys, color='#2c3e50', s=45, marker='o', edgecolors='white', linewidth=0.8, label='Vertices $V_i$',
                   zorder=3)
        label_offset = 3.75

        index = 0
        for (vx, vy) in self.vertices:
            l = f"V {index}"
            index += 1
            ax.annotate(f"${l}$", xy=(vx, vy), xytext=(label_offset, label_offset), textcoords='offset points',
                        fontsize=11, fontweight='bold')

        ax.scatter(self.origin[0], self.origin[1], color='#e74c3c', s=70, marker='o', linewidth=1.5,
                   label=r'Origin $O(0,0)$', zorder=4)
        if False and self.barycenter is not None: ax.scatter(self.barycenter[0], self.barycenter[1], color='green',
                                                             s=70, marker='o',
                                                             linewidth=1.5, label=r'Center', zorder=4)

        ax.set_title("Geometric Analysis of the Closed Form Domain", pad=15)
        ax.set_xlabel(r"Transverse Coordinate $x$ [mm]")
        ax.set_ylabel(r"Longitudinal Coordinate $y$ [mm]")

        ax.set_aspect('equal', adjustable='datalim')

        ax.grid(True, which='major', linestyle='-', alpha=0.4, color='gray')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))  # Stacca leggermente l'asse Y
        ax.spines['bottom'].set_position(('outward', 10))  # Stacca leggermente l'asse X

        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                           ncol=3, frameon=True, fancybox=True, shadow=False,
                           edgecolor='#dcdcdc', facecolor='white')

        legend.get_frame().set_linewidth(0.5)

        if show: plt.show()
        return ax