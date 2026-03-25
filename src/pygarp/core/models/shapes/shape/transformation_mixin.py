from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy import typing as npt
from shapely import affinity, Polygon, Point

from pygarp.core.models.commons import Caches, TransformationRef


class TransformationRequirements(ABC):
    @property
    @abstractmethod
    def polygon(self) -> Polygon:
        pass

    @abstractmethod
    def _set_polygon(self, polygon: Polygon):
        pass

    @property
    @abstractmethod
    def origin(self) -> npt.NDArray[np.float64]:
        pass

    @origin.setter
    @abstractmethod
    def origin(self, origin: npt.NDArray[np.float64]):
        pass

    @property
    @abstractmethod
    def barycenter(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def reset(self, resets: Caches):
        pass

    @abstractmethod
    def reset_all(self):
        pass


class TranslationMixin(TransformationRequirements, ABC):
    def translate(self, x_off: float = 0.0, y_off: float = 0.0) -> Self:
        self._set_polygon(affinity.translate(self.polygon, xoff=x_off, yoff=y_off))
        self.origin = self.origin + np.array([x_off, y_off])

        self.reset(["bounds", "barycenter", "closure"])
        return self


class RotationMixin(TransformationRequirements, ABC):
    def rotate(self, angle: float = 0.0, ref: TransformationRef = "origin") -> Self:
        ref_origin = self.origin if ref == "origin" else self.barycenter
        self._set_polygon(
            affinity.rotate(
                self.polygon, angle=angle, origin=(ref_origin[0], ref_origin[1])
            )
        )

        if ref == "center":
            orig_point = affinity.rotate(
                Point(*self.origin), angle=angle, origin=(ref_origin[0], ref_origin[1])
            )
            self.origin = np.array([orig_point.x, orig_point.y], dtype=np.float64)

        self.reset(["bounds", "barycenter", "closure"])
        return self


class ScaleMixin(TransformationRequirements, ABC):
    def scale(
        self,
        x_fact: float = 1.0,
        y_fact: float = 1.0,
        ref: TransformationRef = "origin",
    ) -> Self:
        ref_origin = self.origin if ref == "origin" else self.barycenter

        self._set_polygon(
            affinity.scale(
                self.polygon,
                xfact=x_fact,
                yfact=y_fact,
                origin=(ref_origin[0], ref_origin[1]),
            )
        )

        if ref == "center":
            orig_point = affinity.scale(
                Point(*self.origin),
                xfact=x_fact,
                yfact=y_fact,
                origin=(ref_origin[0], ref_origin[1]),
            )
            self.origin = np.array([orig_point.x, orig_point.y], dtype=np.float64)

        self.reset_all()
        return self


class TransformationMixin(TranslationMixin, RotationMixin, ScaleMixin, ABC):
    pass
