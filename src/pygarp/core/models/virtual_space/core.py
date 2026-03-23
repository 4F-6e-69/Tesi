import warnings
from typing import Self

import numpy as np
from numpy import typing as npt

from pygarp.core.models.commons import Eps, ArrayLike
from pygarp.core.models.validators import (
    validate_nd_coordinates,
    validate_array_of_nd_coordinates,
)


class WorkingSpace:
    def __init__(
        self,
        origin: ArrayLike,
        x_axis: ArrayLike,
        y_axis: ArrayLike,
        z_axis: ArrayLike,
        _skip=False,
    ):
        if _skip:
            o = np.zeros(3) if origin is None else origin

            x = np.asarray([1, 0, 0], dtype=np.float64) if x_axis is None else x_axis
            y = np.asarray([0, 1, 0], dtype=np.float64) if y_axis is None else y_axis
            z = np.asarray([0, 0, 1], dtype=np.float64) if z_axis is None else z_axis
        else:
            o: npt.NDArray[np.float64] = (
                np.zeros(3) if origin is None else validate_nd_coordinates(origin, 3)
            )

            x_raw: npt.NDArray[np.float64] = (
                np.asarray([1, 0, 0], dtype=np.float64)
                if x_axis is None
                else validate_nd_coordinates(x_axis, 3)
            )
            norm_x: float = np.linalg.norm(x_raw, axis=0)
            y_raw: npt.NDArray[np.float64] = (
                np.asarray([0, 1, 0], dtype=np.float64)
                if y_axis is None
                else validate_nd_coordinates(y_axis, 3)
            )
            norm_y: float = np.linalg.norm(y_raw, axis=0)
            z_raw: npt.NDArray[np.float64] = (
                np.asarray([0, 0, 1], dtype=np.float64)
                if z_axis is None
                else validate_nd_coordinates(z_axis, 3)
            )
            norm_z: float = np.linalg.norm(z_raw, axis=0)

            if norm_x < Eps.eps08 or norm_y < Eps.eps08 or norm_z < Eps.eps08:
                raise ValueError(
                    "Gli assi forniti non possono essere vettori nulli (lunghezza pari a zero)."
                )

            x = x_raw / norm_x
            y = y_raw / norm_y
            z = z_raw / norm_z

            if not WorkingSpace.check_axes(x, y, z, Eps.eps08):
                raise ValueError(
                    "Gli assi forniti (x_axis, y_axis, z_axis) non sono reciprocamente ortogonali."
                )

        self._origin = o

        self._x_axis = x
        self._y_axis = y
        self._normal = z
        self._R = np.stack([x, y, z])

    @classmethod
    def new_space_from_three_points(
        cls, origin: ArrayLike, x_hint: ArrayLike, plane_hint: ArrayLike
    ) -> "WorkingSpace":
        o: npt.NDArray[np.float64] = (
            np.zeros(3) if origin is None else validate_nd_coordinates(origin, 3)
        )
        xh: npt.NDArray[np.float64] = (
            np.asarray([1, 0, 0], dtype=np.float64)
            if x_hint is None
            else validate_nd_coordinates(x_hint, 3)
        )
        ph: npt.NDArray[np.float64] = (
            np.asarray([0, 1, 0], dtype=np.float64)
            if plane_hint is None
            else validate_nd_coordinates(plane_hint, 3)
        )

        v1 = xh - o
        v2 = ph - o
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 < Eps.eps08 or len_v2 < Eps.eps08:
            raise ValueError(
                "Impossibile definire gli assi: 'x_hint' o 'plane_hint' coincidono con l'origine."
            )

        ux = v1 / len_v1
        n_raw = np.cross(ux, v2)
        n_norm = np.linalg.norm(n_raw)

        if n_norm < Eps.eps08:
            raise ValueError(
                "Impossibile definire un piano: i tre punti forniti sono collineari."
            )

        n = n_raw / n_norm
        uy_raw = np.cross(n, ux)
        uy = uy_raw / np.linalg.norm(uy_raw)

        if cls.check_axes(ux, uy, n, Eps.eps08):
            return cls(origin=o, x_axis=ux, y_axis=uy, z_axis=n, _skip=True)
        raise RuntimeError(
            "Errore fatale interno: gli assi generati dai tre punti non risultano ortogonali."
        )

    @classmethod
    def new_space_from_normal(
        cls, origin: ArrayLike, normal: ArrayLike, x_hint: ArrayLike
    ) -> "WorkingSpace":
        o: npt.NDArray[np.float64] = (
            np.zeros(3) if origin is None else validate_nd_coordinates(origin, 3)
        )
        n_raw: npt.NDArray[np.float64] = (
            np.asarray([0, 0, 1], dtype=np.float64)
            if normal is None
            else validate_nd_coordinates(normal, 3)
        )
        xh: npt.NDArray[np.float64] = (
            np.asarray([1, 0, 0], dtype=np.float64)
            if x_hint is None
            else validate_nd_coordinates(x_hint, 3)
        )

        n_len = np.linalg.norm(n_raw)
        if n_len < Eps.eps08:
            raise ValueError(
                "Impossibile definire il piano: la normale fornita è un vettore nullo."
            )
        n = n_raw / n_len
        x_raw = xh - o
        x_len = np.linalg.norm(x_raw)
        if x_len < Eps.eps08:
            raise ValueError(
                "Impossibile definire l'orientamento: 'x_hint' coincide con l'origine."
            )
        ux_hint = x_raw / x_len
        y_raw = np.cross(n, ux_hint)
        y_len = np.linalg.norm(y_raw)
        if y_len < Eps.eps08:
            raise ValueError(
                "Impossibile definire l'orientamento: 'x_hint' è perfettamente parallelo alla normale."
            )

        uy = y_raw / y_len
        x_ortho_raw = np.cross(uy, n)
        ux = x_ortho_raw / np.linalg.norm(x_ortho_raw)

        if cls.check_axes(ux, uy, n, Eps.eps08):
            return cls(origin=o, x_axis=ux, y_axis=uy, z_axis=n, _skip=True)

        raise RuntimeError(
            "Errore fatale interno: gli assi generati non risultano ortogonali."
        )

    @classmethod
    def new_space_from_two_straight(
        cls,
        x_axis: ArrayLike,
        x_hint: ArrayLike,
        y_axis: ArrayLike,
        y_hint: ArrayLike,
        verbose: bool = False,
    ) -> "WorkingSpace":
        x_raw: npt.NDArray[np.float64] = (
            np.asarray([1, 0, 0], dtype=np.float64)
            if x_axis is None
            else validate_nd_coordinates(x_axis, 3)
        )
        y_raw: npt.NDArray[np.float64] = (
            np.asarray([0, 1, 0], dtype=np.float64)
            if y_axis is None
            else validate_nd_coordinates(y_axis, 3)
        )
        xh: npt.NDArray[np.float64] = (
            np.zeros(3) if x_hint is None else validate_nd_coordinates(x_hint, 3)
        )
        yh: npt.NDArray[np.float64] = (
            np.zeros(3) if y_hint is None else validate_nd_coordinates(y_hint, 3)
        )

        x_len = np.linalg.norm(x_raw)
        y_len = np.linalg.norm(y_raw)
        if x_len < Eps.eps08 or y_len < Eps.eps08:
            raise ValueError(
                "I vettori di direzione 'x_axis' o 'y_axis' non sono validi (lunghezza nulla)."
            )

        x_dir = x_raw / x_len
        y_dir = y_raw / y_len
        cross_uv = np.cross(x_dir, y_dir)
        norm_cross = np.linalg.norm(cross_uv)

        if norm_cross < Eps.eps08:
            dp = yh - xh

            rejection = dp - np.dot(dp, x_dir) * x_dir
            dist_between_lines = np.linalg.norm(rejection)

            if dist_between_lines < Eps.eps08:
                raise ValueError(
                    "Le rette fornite sono collineari (coincidenti): impossibile definire un piano unico."
                )

            n_raw = np.cross(x_dir, dp)
            n = n_raw / np.linalg.norm(n_raw)

            center = xh
            final_ux = x_dir
            final_uy = np.cross(n, final_ux)

        else:
            if np.allclose(xh, yh, atol=Eps.eps12):
                center = xh
            else:
                a = np.stack([x_dir, -y_dir], axis=1)
                b = yh - xh

                t_s, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
                t, s_param = t_s

                if verbose:
                    print(f" > Parametri calcolati: t={t:.4f}, s={s_param:.4f}")

                point_on_ux = xh + t * x_dir
                point_on_uy = yh + s_param * y_dir

                dist_mismatch = np.linalg.norm(point_on_ux - point_on_uy)

                if dist_mismatch > Eps.eps06:
                    raise ValueError(
                        f"Le rette sono sghembe e non complanari. Distanza minima: {dist_mismatch:.2e}"
                    )

                center = point_on_ux + (point_on_uy - point_on_ux) / 2.0

            n = cross_uv / norm_cross
            final_ux = x_dir
            final_uy = np.cross(n, final_ux)

        if cls.check_axes(final_ux, final_uy, n, Eps.eps06):
            return cls(
                origin=center, x_axis=final_ux, y_axis=final_uy, z_axis=n, _skip=True
            )

        raise RuntimeError(
            "Errore fatale interno: gli assi generati non risultano ortogonali."
        )

    @property
    def x_axis(self) -> np.typing.NDArray[np.float64]:
        return self._x_axis

    @property
    def y_axis(self) -> np.typing.NDArray[np.float64]:
        return self._y_axis

    @property
    def z_axis(self) -> np.typing.NDArray[np.float64]:
        return self._normal

    @property
    def normal(self) -> np.typing.NDArray[np.float64]:
        return self._normal

    @property
    def space_matrix(self) -> np.typing.NDArray[np.float64]:
        return self._R

    @property
    def space_transposed_matrix(self) -> np.typing.NDArray[np.float64]:
        return np.ascontiguousarray(self._R.T)

    @property
    def axes(self) -> np.typing.NDArray[np.float64]:
        if WorkingSpace.check_axes(self.x_axis, self.y_axis, self._normal, Eps.eps10):
            return self._R

        raise RuntimeError(
            "Stato corrotto: x_axis, y_axis e z_axis non sono più mutuamente ortogonali."
        )

    @property
    def origin(self) -> np.typing.NDArray[np.float64]:
        return self._origin

    def global_to_local(self, points: ArrayLike) -> np.typing.NDArray[np.float64]:
        points_array = validate_array_of_nd_coordinates(points, 3)
        shifted_points = points_array - self.origin
        return shifted_points @ self.space_transposed_matrix

    def local_to_global(self, points: ArrayLike) -> np.typing.NDArray[np.float64]:
        local_points = validate_array_of_nd_coordinates(points, 3)
        rotated_points = local_points @ self.space_matrix
        return rotated_points + self.origin

    def space_to_local(
        self,
        points: ArrayLike,
        *,
        space: Self | None = None,
        matrix: np.typing.NDArray[np.float64] | None = None,
        center: ArrayLike | None = None,
    ) -> np.typing.NDArray[np.float64]:

        if space is not None:
            if not isinstance(space, WorkingSpace):
                raise TypeError(
                    "L'argomento 'space' deve essere un'istanza di WorkingSpace."
                )

            space.is_valid()
            global_points = space.local_to_global(points)
            return self.global_to_local(global_points)

        if matrix is None and center is None:
            raise ValueError(
                "Parametri insufficienti: devi fornire 'space', oppure 'matrix' e/o 'center'."
            )

        if matrix is None:
            matrix = np.eye(3, dtype=np.float64)
        else:
            if not WorkingSpace.validate_matrix(matrix):
                warnings.warn(
                    "La matrice fornita (other_matrix) introduce deformazioni (non è ortonormale)."
                )

        center = (
            np.zeros(3, dtype=np.float64)
            if center is None
            else validate_nd_coordinates(center, 3)
        )

        points_array = validate_array_of_nd_coordinates(points, 3)
        global_points = points_array @ matrix + center
        return self.global_to_local(global_points)

    def invert_x(self) -> Self:
        self._x_axis = -self.x_axis
        self.__update_matrix()
        return self

    def invert_y(self) -> Self:
        self._y_axis = -self.y_axis
        self.__update_matrix()
        return self

    def invert_z(self) -> Self:
        self._normal = -self.normal
        self.__update_matrix()
        return self

    def invert_normal(self) -> Self:
        return self.invert_z()

    def invert_space(self) -> Self:
        self._x_axis = -self.x_axis
        self._y_axis = -self.y_axis
        self._normal = -self.normal
        self.__update_matrix()
        return self

    def invert_plane(self) -> Self:
        self._x_axis = -self.x_axis
        self._y_axis = -self.y_axis
        self.__update_matrix()
        return self

    def exchange_plane(self) -> Self:
        temp_x = self.x_axis
        self._x_axis = self.y_axis
        self._y_axis = temp_x
        self.__update_matrix()
        return self

    def __update_matrix(self) -> None:
        self._R = np.stack([self.x_axis, self.y_axis, self.normal])

    @staticmethod
    def check_axes(
        ux: np.typing.NDArray[np.float64],
        uy: np.typing.NDArray[np.float64],
        uz: np.typing.NDArray[np.float64],
        tolerance,
    ):
        space_matrix = np.stack([ux, uy, uz], dtype=np.float64)
        return bool(
            np.allclose(space_matrix.T @ space_matrix, np.eye(3), atol=tolerance)
        )

    def is_valid(self) -> bool:
        is_orthonormal = WorkingSpace.validate_matrix(self.space_matrix)
        if not is_orthonormal:
            warnings.warn(
                "Il sistema di riferimento introduce deformazioni (la matrice non è perfettamente ortonormale)."
            )
            return False

        is_right_handed = np.isclose(
            np.linalg.det(self.space_matrix), 1.0, atol=Eps.eps08
        )
        if not is_right_handed:
            warnings.warn(
                "Il sistema di riferimento non è destrorso puro (potrebbe contenere una riflessione degli assi)."
            )
            return False

        return True

    @staticmethod
    def validate_space(space):
        if not isinstance(space, WorkingSpace):
            raise TypeError(
                "L'oggetto fornito non è un'istanza valida di WorkingSpace."
            )

        if not space.is_valid():
            raise ValueError(
                "Lo spazio fornito risulta corrotto o non valido (controlla i warning per i dettagli)."
            )

    @staticmethod
    def validate_matrix(space_matrix: np.typing.NDArray[np.float64]):
        if not isinstance(space_matrix, np.ndarray) or space_matrix.shape != (3, 3):
            raise ValueError(f"La matrice deve essere un ndarray 3x3.")

        return bool(
            np.allclose(space_matrix.T @ space_matrix, np.eye(3), atol=Eps.eps08)
        )
