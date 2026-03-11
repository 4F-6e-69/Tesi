import warnings

import numpy as np
from numpy import typing as nptyping

from shapely.geometry import Polygon, Point
from shapely import affinity

from src.utils import ArrayLike, Resets, Ref
from src.utils import validate_array_of_2d_coordinates, validate_2d_coordinates

class Shape:
    def __init__(self, points: ArrayLike, origin: ArrayLike = None, **kwargs) -> None:
        __skip = kwargs.get("__skip", False)
        __calc_closure = kwargs.get("__closure", None)

        # validazione
        if not __skip:
            validated_points = validate_array_of_2d_coordinates(points)
            points = points if validated_points is None else validated_points

            if origin is not None:
                validated_origin = validate_2d_coordinates(origin)
                origin = origin if validated_origin is None else validated_origin

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
        self._barycenter: nptyping.NDArray[np.float64] | None = None

    @property
    def shapely(self) -> Polygon | None:
        if self._shapely_shape is None:
            warnings.warn("L'oggetto Shapely non è stato ancora inizializzato (output None)")
            return None
        return self._shapely_shape

    @property
    def origin(self) -> nptyping.NDArray[np.float64] | None:
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
    def barycenter(self) -> nptyping.NDArray[np.float64] | None:
        if self._barycenter is None:
            self._barycenter = np.asarray([self.shapely.centroid.x, self.shapely.centroid.y], dtype=np.float64)
        return self._barycenter

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
            elif target == "area":
                self._area = None
            elif target == "bounds":
                self._bounds = None
            elif target == "barycenter":
                self._barycenter = None

    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> 'Shape':
        self._shapely_shape = affinity.translate(self.shapely, xoff=x_off, yoff=y_off)

        if self._origin is not None:
            self._origin = self._origin + np.array([x_off, y_off])

        self.reset(["bounds", "barycenter"])
        return self
    def rotate(self, angle: float = 0.0, ref: Ref = "origin") -> 'Shape':
        ref_origin = self.origin if ref == "origin" else self.barycenter
        self._shapely_shape = affinity.rotate(self.shapely, angle=angle, origin=(ref_origin[0], ref_origin[1]))

        if ref == "barycenter":
            orig_point = affinity.rotate(Point(self._origin), angle=angle, origin=(ref_origin[0], ref_origin[1]))
            self._origin = np.array([orig_point.x, orig_point.y], dtype=np.float64)

        self.reset(["bounds", "barycenter"])
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
        return self

    def __str__(self):
        return self._shapely_shape.__str__().split(" ")[0]
    def __repr__(self):
        return self._shapely_shape.__repr__()