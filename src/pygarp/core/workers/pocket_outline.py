import numpy as np
from numpy import typing as npt

from scipy.interpolate import interp1d

from typing import Tuple
from pygarp.core.models.commons import ArrayLike


def calc_step_outline(points: npt.NDArray[np.float64], gamma: float):
    path, dim = _init_path(points)
    tan = _calc_tangent(path)
    gamma_rad = -np.deg2rad(gamma)
    alpha = np.arctan2(tan[:, 1], tan[:, 0])
    ca, sa = np.cos(alpha), np.sin(alpha)
    cg, sg = np.cos(gamma_rad), np.sin(gamma_rad)

    R = np.zeros((dim, 3, 3))
    R[:, 0, 0] = ca
    R[:, 1, 0] = sa
    R[:, 0, 1] = -sa * cg
    R[:, 1, 1] = ca * cg
    R[:, 2, 1] = sg
    R[:, 0, 2] = sa * sg
    R[:, 1, 2] = -ca * sg
    R[:, 2, 2] = cg

    v = R @ np.array([0, 0, 1])
    stack = np.hstack((path, path + v))
    return np.vstack((stack, stack[0]))


def calc_gradient_outline(points: ArrayLike, gamma: float):
    path, dim = _init_path(points)
    tan = _calc_tangent(path)

    tan_norm = np.linalg.norm(tan, axis=1, keepdims=True)
    tan_norm[tan_norm == 0] = 1e-10
    gamma_rad = np.deg2rad(gamma)
    new_tangent_xy = (-tan / tan_norm) * np.sin(gamma_rad)

    v = np.zeros((dim, 3))
    v[:, :2] = -new_tangent_xy
    v[:, 2] = np.cos(gamma_rad)

    stack = np.hstack((path, path + v))
    return np.vstack((stack, stack[0]))


def _init_path(
    points: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], float]:
    path = (
        np.delete(points, -1, axis=0)
        if np.allclose(points[0], points[-1])
        else points.copy()
    )
    dim_x, dim_y = path.shape

    if dim_y == 2:
        final_path = np.column_stack((path, np.zeros(dim_x)))
    elif dim_y == 3:
        final_path = path
    else:
        if dim_y < 2:
            raise ValueError()

        final_path = path[:, :3]

    return final_path, dim_x


def _calc_tangent(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    prev_p = np.roll(points[:, :2], 1, axis=0)
    next_p = np.roll(points[:, :2], -1, axis=0)
    return next_p - prev_p


def discretize_points(points: npt.NDArray[np.float64], step: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)
    segment_distances = np.sqrt(dx**2 + dy**2)

    cumulative_distance = np.insert(np.cumsum(segment_distances), 0, 0)
    tot_distance = cumulative_distance[-1]

    if tot_distance == 0 or np.isnan(tot_distance):
        return points

    num_of_segments = max(1, int(np.round(tot_distance / step)))
    num_of_points = num_of_segments + 1

    distanze_target = np.linspace(0, tot_distance, num_of_points)

    interp_x = interp1d(cumulative_distance, x, kind="linear")
    interp_y = interp1d(cumulative_distance, y, kind="linear")
    x_equi = interp_x(distanze_target)
    y_equi = interp_y(distanze_target)
    z_0 = np.zeros_like(x_equi)

    return np.column_stack((x_equi, y_equi, z_0))
