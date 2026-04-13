from typing import Tuple

import numpy as np
from numpy import typing as npt
from shapely.geometry import Polygon

from pygarp.core.models.commons import EpsConfig, Eps
from pygarp.core.models.commons import ArrayLike, Caches
from pygarp.core.models.utils import sort_by_tolerance_2d_array
from pygarp.core.models.validators import (
    validate_nd_coordinates,
    validate_array_of_nd_coordinates,
)

from pygarp.core.models.shapes.shape.transformation_mixin import TransformationMixin
from pygarp.core.models.shapes.shape.discretization_mixin import DiscretizationMixin


class Shape(DiscretizationMixin, TransformationMixin):
    def __init__(
        self,
        points: ArrayLike,
        origin: ArrayLike = None,
        *,
        identifier: str | None = None,
        name: str | None = None,
        description: str | None = None,
        _skip: bool = False,
        assume_sort: bool = False,
        eps: EpsConfig | float = Eps.eps10,
    ):

        if _skip:
            points_array = points
            origin_array = (
                origin if origin is not None else np.zeros(2, dtype=np.float64)
            )
        else:
            points_array: npt.NDArray[np.float64] = validate_array_of_nd_coordinates(
                points, 2
            )
            origin_array: npt.NDArray[np.float64] = (
                validate_nd_coordinates(origin, 2)
                if origin is not None
                else np.zeros(2, dtype=np.float64)
            )

        if not assume_sort:
            points_array = sort_by_tolerance_2d_array(
                points_array, decimals=abs(int(np.log10(eps)))
            )

        shape: Polygon = Polygon(points_array)
        self._polygon: Polygon = shape

        centroid_x, centroid_y = shape.centroid.x, shape.centroid.y
        centroid = np.array([centroid_x, centroid_y], dtype=np.float64)
        self._origin: npt.NDArray[np.float64] = origin_array
        self._origin_is_centroid: bool = np.allclose(origin_array, centroid, atol=eps)
        self._barycenter: npt.NDArray[np.float64] | None = centroid

        self._area: float | None = None
        self._length: float | None = None
        self._bounds: Tuple[float, float, float, float] | None = None

        self._id: str | None = identifier
        self._name: str | None = name
        self._description: str | None = description

        super().__init__()

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        if self._origin is None:
            self._origin = np.zeros(2, dtype=np.float64).flatten()
        return self._origin

    @property
    def origin_is_centroid(self) -> bool:
        return self._origin_is_centroid

    @origin.setter
    def origin(self, origin: ArrayLike) -> None:
        self.set_origin(origin)

    def set_origin(self, origin: ArrayLike, *, eps: EpsConfig = Eps.eps10) -> None:
        origin_array = validate_nd_coordinates(origin, 2)

        if np.allclose(origin_array, self.barycenter, atol=eps):
            self._origin_is_centroid = True
        else:
            self._origin_is_centroid = False

        self._origin = origin_array

    @property
    def barycenter(self) -> npt.NDArray[np.float64]:
        if self._barycenter is None:
            self._barycenter = np.array(
                [self._polygon.centroid.x, self._polygon.centroid.y], dtype=np.float64
            )
        return self._barycenter

    @property
    def ccw(self) -> bool:
        return self.polygon.exterior.is_ccw

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self._polygon.area
        return self._area

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self._polygon.length
        return self._length

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        if self._bounds is None:
            self._bounds = self._polygon.bounds
        return self._bounds

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    def _set_polygon(self, polygon: Polygon):
        self._polygon = polygon
        self.reset_all()

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        x, y = self._polygon.exterior.xy
        return np.column_stack((x, y))

    @property
    def open_control_points(self) -> npt.NDArray[np.float64]:
        return np.delete(self.control_points, -1, axis=0)

    def reset_all(self):
        # Cache Base
        self._area = None
        self._length = None
        self._bounds = None
        self._barycenter = None

        # Svuota anche la Cache di Discretizzazione (Fondamentale se la forma cambia!)
        self._closure = None
        self._step_max = None
        self._step_min = None
        self._sure_steps = None

    def reset(self, resets: Caches):
        if resets is None:
            self.reset_all()
            return

        for target in resets:
            if target == "all":
                self.reset_all()
                return
            elif target == "length":
                self._length = None
            elif target == "area":
                self._area = None
            elif target == "bounds":
                self._bounds = None
            elif target == "barycenter":
                self._barycenter = None
            elif target == "closure":
                self._closure = None

    @property
    def id(self):
        return self._id

    def _set_id(self, identifier: str):
        self._id = str(identifier)

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str | None):
        self._name = name

    @property
    def description(self) -> str | None:
        return self._description

    @description.setter
    def description(self, description: str | None):
        self._description = description

    def __str__(self) -> str:
        """Rappresentazione leggibile, concisa e user-friendly."""
        ident = (
            f"'{self.name}'"
            if self.name
            else (f"ID:{self.id}" if self.id else "Unnamed")
        )
        n_vertices = len(self.polygon.exterior.coords) - 1

        return f"Shape {ident} (Poligono a {n_vertices} vertici, Area: {self.area:.2f})"

    def __repr__(self) -> str:
        """Rappresentazione tecnica e dettagliata per il debug e i log."""
        n_vertices = len(self.polygon.exterior.coords) - 1
        b = self.bounds
        bounds_str = (
            f"({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f})" if b else "None"
        )
        orig = self.origin
        origin_str = f"[{orig[0]:.2f}, {orig[1]:.2f}]"
        step_str = f"{self._step:.3f}" if self._step is not None else "Non calcolato"

        return (
            f"<Shape(id={self.id}, name={self.name!r}) | "
            f"Vertici: {n_vertices} | "
            f"Origine: {origin_str} | "
            f"Bounds: {bounds_str} | "
            f"Area: {self.area:.2f} | "
            f"Passo: {step_str}>"
        )
