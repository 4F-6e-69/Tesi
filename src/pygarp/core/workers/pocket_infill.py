from typing import List

import numpy as np
from numpy import typing as npt
from shapely import MultiPolygon

from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.affinity import rotate

from pygarp.core.workers.pocket_outline import _discretize_points


def linear_rect_fill_path(
    intersections: npt.NDArray[np.float64], sure_quote: float
) -> npt.NDArray[np.float64]:
    num_segments, num_points, num_coords = intersections.shape
    if num_points != 2 or num_coords != 3:
        raise ValueError(
            f"Atesi segmenti con 2 punti a 3 coordinate, ricevuto {num_points}x{num_coords}"
        )

    off = np.asarray([0, 0, sure_quote], dtype=np.float64)
    v = np.zeros((num_segments, 4, num_coords), dtype=np.float64)
    v[:, 0, :] = intersections[:, 0, :] + off
    v[:, 1:3, :] = intersections
    v[:, 3, :] = intersections[:, 1, :] + off

    return v.reshape(-1, 3)


def concentri_plot_fill_path(
    polygons: List[Polygon],
    base_step: float,
    sure_quote: float,
    interpolate: bool = False,
) -> npt.NDArray[np.float64]:
    if not polygons:
        return np.asarray([], dtype=np.float64)
    points_list = []
    exterior_length = polygons[0].exterior.length

    for index, poly in enumerate(polygons):
        coords = np.column_stack(poly.exterior.xy)

        if interpolate:
            step_factor = poly.exterior.length / exterior_length
            current_step = base_step * step_factor * (1.2**index)

            polygon_points = _discretize_points(coords, current_step)

            off = np.asarray([[0.0, 0.0, sure_quote]], dtype=np.float64)

            approach_point = polygon_points[0:1, :] + off
            retract_point = polygon_points[-1:, :] + off

            points_list.extend([approach_point, polygon_points, retract_point])

        else:
            if coords.shape[1] == 2:
                coords = np.column_stack((coords, np.zeros(coords.shape[0])))
            points_list.append(coords)

    return np.vstack(points_list).astype(np.float64)


def calc_linear_intersection(
    poly: Polygon, alpha: float, off: float
) -> npt.NDArray[np.float64]:
    inner = []
    centroid = poly.centroid
    rotated_polygon = rotate(poly, -alpha, origin=centroid)

    x_0, y_0, x_f, y_f = rotated_polygon.exterior.bounds
    n = int(np.round((x_f - x_0) / off))
    step = (x_f - x_0) / n if n > 0 else off

    for i in range(n):
        x_curr = x_0 + step / 2 + i * step
        line = (
            LineString([(x_curr, y_0), (x_curr, y_f)])
            if np.isclose(i % 2, 0, atol=1e-8)
            else LineString([(x_curr, y_f), (x_curr, y_0)])
        )
        intersections = rotated_polygon.intersection(line)
        if intersections.is_empty:
            continue

        geometries = (
            intersections.geoms
            if isinstance(intersections, MultiLineString)
            else [intersections]
        )
        for segmento in geometries:
            real_segment = rotate(segmento, alpha, origin=centroid)
            x, y = real_segment.xy

            inner_segment = np.column_stack((x, y, np.zeros(len(x))))
            inner.append(inner_segment)

    if not inner:
        return np.asarray([], dtype=np.float64)
    return np.asarray(inner, dtype=np.float64)


def calc_concentric_shapes(
    poly: Polygon,
    cycle: int,
    off: float,
    off_0: float,
    first: bool = False,
    flatter: bool = False,
) -> List[Polygon]:
    current_poly = poly.buffer(-off_0, join_style="mitre")
    poly_succession = [] if first else [current_poly]

    for i in range(cycle):
        current_poly = current_poly.buffer(-off, join_style="mitre")
        if current_poly.is_empty:
            break

        if current_poly.geom_type == "Polygon":
            polygons = [current_poly]
        elif isinstance(current_poly, MultiPolygon):
            polygons = current_poly.geoms
        else:
            break

        if flatter:
            for p in polygons:
                poly_succession.append(p)
        else:
            poly_succession.append(polygons)

    if not poly_succession:
        return []

    return poly_succession
