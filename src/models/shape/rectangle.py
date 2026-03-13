import numpy as np
from numpy import typing as npt

from src.models.shape.generic_shape import Shape
from src.utils import ArrayLike, Eps, Ref
from src.utils import validate_2d_coordinates
from src.models.shape.generic_polygon import GenericPolygon


class Rectangle(GenericPolygon):
    def __init__(self, a: ArrayLike = None, b: ArrayLike = None, c: ArrayLike = None, d: ArrayLike = None, **kwargs):
        __points_array = kwargs.pop('__points_array', None)

        if __points_array is None:
            def _get_np_pt(pt):
                val = validate_2d_coordinates(pt)
                return np.asarray(pt if val is None else val, dtype=np.float64)

            A, B, C, D = _get_np_pt(a), _get_np_pt(b), _get_np_pt(c), _get_np_pt(d)

            w_vec = B - A
            h_vec = C - B
            w_backup_vec = D - C
            h_backup_vec = A - D

            w = np.linalg.norm(w_vec)
            h = np.linalg.norm(h_vec)
            w_backup = np.linalg.norm(w_backup_vec)
            h_backup = np.linalg.norm(h_backup_vec)

            if not np.isclose(w, w_backup, atol=Eps.eps12) or not np.isclose(h, h_backup, atol=Eps.eps12):
                raise ValueError("Errore: i lati opposti non sono uguali. I punti non formano un parallelogramma.")

            if not np.isclose(np.dot(w_vec, h_vec), 0.0, atol=Eps.eps06):
                raise ValueError("Errore: gli angoli tra i lati non sono di 90 gradi.")

            self._w = float(w)
            self._h = float(h)
            __points_array = np.asarray([A, B, C, D], dtype=np.float64)

        else:
            self._w = float(np.linalg.norm(__points_array[1] - __points_array[0]))
            self._h = float(np.linalg.norm(__points_array[2] - __points_array[1]))

        super().__init__(points=__points_array, origin=np.zeros(2).flatten(), __skip=True)
    @classmethod
    def new_rect(cls, o: ArrayLike, width: float, height: float) -> 'Rectangle':
        new_O = validate_2d_coordinates(o)
        D = o if new_O is None else new_O

        if np.isclose(width, 0.0, atol=Eps.eps04) or np.isclose(height, 0.0, atol=Eps.eps04):
            raise ValueError("Errore nella costruzione del rettangolo: base e altezza troppo vicine allo 0.")

        vertices_matrix = np.asarray([
            [0, height],
            [width, height],
            [width, 0],
            [0, 0],
        ], dtype=np.float64)

        points_array = vertices_matrix + D
        return cls(__points_array=points_array)

    @property
    def w(self) -> float:
        return self._w
    @property
    def h(self) -> float:
        return self._h

    @property
    def theoretical_area(self) -> float:
        return self.w * self.h
    @property
    def theoretical_length(self) -> float:
        return (self.w + self.h) * 2

    def scale(self, x_fact: float = 1.0, y_fact: float = 1.0, ref: Ref = "origin") -> 'Rectangle':
        super().scale(x_fact, y_fact, ref)

        vertices = np.array(self.shapely.exterior.coords[:-1])
        A, B, C = vertices[0], vertices[1], vertices[2]
        w_vec = B - A
        h_vec = C - B

        if not np.isclose(np.dot(w_vec, h_vec), 0.0, atol=1e-6):
            raise ValueError(
                "Attenzione: hai scalato in modo non uniforme un rettangolo ruotato! "
                "Questo deforma gli angoli e la forma non è più un rettangolo valido."
            )

        self._w = float(np.linalg.norm(w_vec))
        self._h = float(np.linalg.norm(h_vec))
        return self