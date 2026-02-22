import numpy as np

class WorkingSpace(object):
    EPS_05 = 1e-5
    EPS_06 = 1e-6
    EPS_08 = 1e-8
    EPS_10 = 1e-10
    EPS_12 = 1e-12
    EPS_13 = 1e-13
    EPS_15 = 1e-15
    EPS_16 = np.finfo(np.float64).eps

    def __init__(self, origin=(0, 0, 0), x_axis=(1, 0, 0), y_axis=(0, 1, 0), z_axis=(0, 0, 1), skip = False):
        if not skip and not WorkingSpace.check_axes(x_axis, y_axis, z_axis, WorkingSpace.EPS_12):
            raise AttributeError('x_axis, y_axis and z_axis are not compatible')

        self._origin = origin

        x = np.asarray(x_axis, dtype=np.float64) / np.linalg.norm(x_axis)
        y = np.asarray(y_axis, dtype=np.float64) / np.linalg.norm(y_axis)
        z = np.asarray(z_axis, dtype=np.float64) / np.linalg.norm(z_axis)

        self._x_axis = x
        self._y_axis = y
        self._normal = z

        self._R = np.stack([
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64)
        ])
        self._R_T = np.ascontiguousarray(self._R.T)

        self.tolerance = WorkingSpace.EPS_10

    @classmethod
    def new_space_from_three_points(cls, origin: np.typing.ArrayLike[np.float64], x_hint: np.typing.ArrayLike[np.float64], plane_hint: np.typing.ArrayLike[np.float64]) -> object:
        o = np.asarray(origin, dtype=np.float64)
        xh = np.asarray(x_hint, dtype=np.float64)
        ph = np.asarray(plane_hint, dtype=np.float64)

        v1 = xh - o
        v2 = ph - o
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 < WorkingSpace.EPS_12 or len_v2 < WorkingSpace.EPS_12:
            raise ValueError('x_hint and plane_hint are not compatible with origin')

        ux = v1 / len_v1

        n_raw = np.cross(ux, v2)
        n_norm = np.linalg.norm(n_raw)
        if n_norm < WorkingSpace.EPS_12:
            raise ValueError('three point are on the same line')
        n = n_raw / n_norm

        uy_raw = np.cross(n, ux)
        uy = uy_raw / np.linalg.norm(uy_raw)

        if cls.check_axes(ux, uy, n, WorkingSpace.EPS_12):
            return cls(origin=o, x_axis=ux, y_axis=uy, z_axis=uy, skip = True)

        raise ValueError()
    @classmethod
    def new_space_from_normal(cls, origin: np.typing.NDArray[np.float64], normal: np.typing.NDArray[np.float64], x_hint: np.typing.NDArray[np.float64]) -> object:
        o = np.array(origin, dtype=np.float64)
        n_raw = np.array(normal, dtype=np.float64)
        xh = np.array(x_hint, dtype=np.float64)

        n_len = np.linalg.norm(n_raw)
        if n_len < WorkingSpace.EPS_12:
            raise ValueError('n is null')
        n = n_raw / n_len

        x_raw = xh - o
        x_len = np.linalg.norm(x_raw)
        if x_len < WorkingSpace.EPS_12:
            raise ValueError('x is null: x_hint and origin must be different')
        ux = x_raw / x_len

        y_raw = np.cross(n, ux)
        y_len = np.linalg.norm(y_raw)
        if y_len < WorkingSpace.EPS_12:
            raise ValueError('y is null: all vector are on the same line')
        uy = y_raw / y_len

        if not cls.check_axes(ux, uy, n, WorkingSpace.EPS_12):
            x_raw = np.cross(uy, n)
            ux = x_raw / np.linalg.norm(x_raw)

        if cls.check_axes(uy, ux, n, WorkingSpace.EPS_12):
            return cls(origin=o, x_axis=ux, y_axis=uy, z_axis=uy, skip = True)
        raise ValueError()
    @ classmethod
    def new_space_from_two_straight(cls, x_axis: np.typing.ArrayLike[np.float64], x_hint: np.typing.ArrayLike[np.float64], y_axis: np.typing.ArrayLike[np.float64], y_hint: np.typing.ArrayLike[np.float64], verbose=False) -> object:
        x_raw = np.array(x_axis, dtype=np.float64)
        y_raw = np.array(y_axis, dtype=np.float64)
        xh = np.array(x_hint, dtype=np.float64)
        yh = np.array(y_hint, dtype=np.float64)

        x_len = np.linalg.norm(x_raw)
        y_len = np.linalg.norm(y_raw)

        if x_len < WorkingSpace.EPS_12 or y_len < WorkingSpace.EPS_12:
            raise ValueError('x_axis o y_axis non sono validi (lunghezza nulla)')

        x_dir = x_raw / x_len
        y_dir = y_raw / y_len

        cross_uv = np.cross(x_dir, y_dir)
        norm_cross = np.linalg.norm(cross_uv)

        if norm_cross < WorkingSpace.EPS_12:
            dp = yh - xh

            rejection = dp - np.dot(dp, x_dir) * x_dir
            dist_between_lines = np.linalg.norm(rejection)

            if dist_between_lines < WorkingSpace.EPS_12:
                raise ValueError('Le rette sono collineari (coincidenti): impossibile definire un piano unico.')

            n_raw = np.cross(x_dir, dp)
            n = n_raw / np.linalg.norm(n_raw)

            center = xh
            final_ux = x_dir
            final_uy = np.cross(n, final_ux)
        else:
            if np.allclose(xh, yh, atol=WorkingSpace.EPS_12):
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

                if dist_mismatch > WorkingSpace.EPS_05:
                    raise ValueError(f"Le rette sono sghembe (non complanari). Distanza: {dist_mismatch:.2e}")

                center = point_on_ux + (point_on_uy - point_on_ux) / 2.0

            n = cross_uv / norm_cross
            final_ux = x_dir

            final_uy = np.cross(n, final_ux)

        if cls.check_axes(final_ux, final_uy, n, WorkingSpace.EPS_05):
            return cls(origin=center, x_axis=final_ux, y_axis=final_uy, z_axis=n, skip=True)

        raise RuntimeError("Fatal Error: Gli assi generati non sono ortogonali")

    @property
    def x_axis(self) -> np.typing.NDArray[np.float64]:
        if self._x_axis is None:
            raise AttributeError('x_axis is not set')

        return self._x_axis
    @property
    def y_axis(self) -> np.typing.NDArray[np.float64]:
        if self._y_axis is None:
            raise AttributeError('y_axis is not set')

        return self._y_axis
    @property
    def z_axis(self) -> np.typing.NDArray[np.float64]:
        if self._normal is None:
            raise AttributeError('z_axis is not set')

        return self._normal
    @property
    def normal(self) -> np.typing.NDArray[np.float64]:
        if self._normal is None:
            raise AttributeError('z_axis is not set')

        return self._normal

    @property
    def space_matrix(self) -> np.typing.NDArray[np.float64]:
        if self._R is None:
            raise AttributeError('R is not set')
        return self._R
    @property
    def space_transposed_matrix(self) -> np.typing.NDArray[np.float64]:
        if self._R is None and self._R_T is None:
            raise AttributeError('R_T is not set')
        return self._R_T

    @property
    def axes(self) -> np.typing.NDArray[np.float64]:
        if WorkingSpace.check_axes(self.x_axis, self.y_axis, self._normal, self.tolerance):
            return np.asarray([self.x_axis, self.y_axis, self.z_axis], dtype=np.float64)

        raise AttributeError('x_axis, y_axis, z_axis are not compatible')
    @property
    def origin(self) -> np.typing.NDArray[np.float64]:
        if self._origin is None:
            raise AttributeError('origin is not set')

        return self._origin

    def global_to_local(self, points: np.typing.ArrayLike[np.float64]) -> np.typing.NDArray[np.float64]:
        points_array = np.asarray(points, dtype=np.float64)
        shifted_points = points_array - self.origin
        return shifted_points @ self.space_matrix
    def local_to_global(self, points: np.typing.ArrayLike[np.float64]) -> np.typing.NDArray[np.float64]:
        local_points = np.asarray(points, dtype=np.float64)
        rotated_points = local_points @ self.space_matrix
        return rotated_points + self.origin
    def space_to_local(self, points: np.typing.ArrayLike[np.float64], **kwargs) -> np.typing.NDArray[np.float64]:
        other_space_matrix = kwargs.get('matrix', None)
        other_space_center = kwargs.get('center', None)
        other_space = kwargs.get('space', None)

        if other_space is not None:
            if not isinstance(other_space, WorkingSpace) or not other_space.is_valid():
                raise AttributeError('other_space is not valid')

            global_points = other_space.local_to_global(points)
            return self.global_to_local(global_points)

        if other_space_matrix is None and other_space_center is None:
            raise AttributeError('space_matrix and center  must not be all None')

        if not WorkingSpace.validate_matrix(other_space_matrix):
            print('matrix introduces deformation, the axes are not orthonormal')

        points_array = np.asarray(points, dtype=np.float64)
        shifted_points = points_array - self.origin
        global_points = shifted_points @ other_space_matrix
        return self.global_to_local(global_points)

    def invert_x(self):
        try:
            self._x_axis = -self.x_axis
            self.__update_matrix()

        except AttributeError as e:
            raise e
    def invert_y(self):
        try:
            self._y_axis = -self.y_axis
            self.__update_matrix()

        except AttributeError as e:
            raise e
    def invert_z(self):
        try:
            self._normal = -self.normal
            self.__update_matrix()

        except AttributeError as e:
            raise e
    def invert_normal(self):
        try:
            self._normal = -self.normal
            self.__update_matrix()

        except AttributeError as e:
            raise e
    def invert_space(self):
        try:
            self._x_axis = -self.x_axis
            self._y_axis = -self.y_axis
            self._normal = -self.normal

            self.__update_matrix()

        except AttributeError as e:
            raise e
    def invert_plane(self):
        try:
            self._x_axis = -self.x_axis
            self._y_axis = -self.y_axis

            self.__update_matrix()

        except AttributeError as e:
            raise e
    def exchange_plane(self):
        try:
            temp_x = self.x_axis
            self._x_axis = self.y_axis
            self._y_axis = temp_x

            self.__update_matrix()

        except AttributeError as e:
            raise e

    def __update_matrix(self):
        self._R = np.stack([
            self.x_axis,
            self.y_axis,
            self.normal
        ])
        self._R_T = np.ascontiguousarray(self._R.T)

    @staticmethod
    def check_axes(ux: np.typing.NDArray[np.float64], uy: np.typing.NDArray[np.float64], uz: np.typing.NDArray[np.float64], tolerance):
        if not isinstance(ux, np.ndarray) or not isinstance(uy, np.ndarray) or not isinstance(uz, np.ndarray):
            raise ValueError('ux, uy and uz must be np.ndarray')

        if np.shape(ux) != (1, 3) or np.shape(uy) != (1, 3) or np.shape(uz) != (1, 3) or np.shape(ux) != (3, 1) or np.shape(uy) != (3, 1) or np.shape(uz) != (3, 1):
            raise ValueError('ux, uy and uz must have same shape = (1 x 3)')

        space_matrix = np.stack([ux, uy, uz], dtype=np.float64)
        is_orthogonal = np.allclose(space_matrix.T @ space_matrix, np.eye(3), atol=tolerance)
        if not is_orthogonal:
            return False

        return True
    def is_valid(self) -> bool:
        if not isinstance(self.origin, np.ndarray) or not isinstance(self.x_axis, np.ndarray) or not isinstance(self.y_axis, np.ndarray) or not isinstance(self.y_axis, np.ndarray) or not isinstance(self.normal, np.ndarray) or not isinstance(self.space_matrix, np.ndarray) or not isinstance(self.space_transposed_matrix, np.ndarray):
            raise AttributeError('some attribute is not valid')

        if self.origin is None or self.x_axis is None or self.y_axis is None or self.normal is None or self.space_matrix is None or self.space_transposed_matrix is None or self.tolerance is None:
            raise AttributeError('some attribute is None')

        if not np.isclose(np.linalg.norm(self.x_axis), 1, self.tolerance) or not np.isclose(np.linalg.norm(self.y_axis), 1, self.tolerance) or not np.isclose(np.linalg.norm(self.normal), 1, self.tolerance):
            print('space introduce deformation')
            return False
            # raise AttributeError('Axis are not normalized')

        check_matrix = np.stack([self.x_axis, self.y_axis, self.normal], dtype=np.float64)
        check_traspose = check_matrix.T

        if not np.allclose(check_matrix, self.space_matrix, self.tolerance) or not np.allclose(check_traspose, self.space_transposed_matrix, self.tolerance):
            print('space introduce deformation')
            return False
            # raise AttributeError('space matrix dont match')

        if not WorkingSpace.validate_matrix(self.space_matrix):
            print('space introduce deformation')
            return False
            # raise AttributeError('space matrix is not valid')

        return True
    @staticmethod
    def validate_space(space):
        if not isinstance(space, WorkingSpace):
            raise AttributeError('space-object is not valid as WorkingSpace')

        space.is_valid()
    @staticmethod
    def validate_matrix(space_matrix: np.typing.NDArray[np.float64]):
        if not isinstance(space_matrix, np.ndarray):
            raise TypeError('space_matrix deve essere un oggetto np.ndarray')

        if space_matrix.shape != (3, 3):
            raise ValueError(f'La matrice deve essere 3x3, trovata forma {space_matrix.shape}')

        is_orthogonal = np.allclose(space_matrix.T @ space_matrix, np.eye(3), atol=WorkingSpace.EPS_08)
        if not is_orthogonal:
            return False

        return True