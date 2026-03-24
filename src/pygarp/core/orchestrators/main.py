from typing import Union, List

import numpy as np
from numpy import typing as npt

from shapely.geometry import Polygon

from pygarp.core.models.shapes.shape.core import Shape
from pygarp.core.models.shapes.closed_spline import ClosedSpline
from pygarp.core.models.shapes.circle import Circle
from pygarp.core.models.shapes.rectangle import Rectangle
from pygarp.core.models.shapes.regular_polygon import RegularPolygon

from pygarp.core.models.virtual_space.core import WorkingSpace

from pygarp.core.models.validators import (
    ShapeConfig,
    SpaceConfig,
    ScarfingConfig,
    RobotConfig,
)
from pygarp.core.workers.pocket_infill import (
    linear_rect_fill_path,
    calc_linear_intersection,
)
from pygarp.core.workers.pocket_infill import (
    concentric_plot_fill_path,
    calc_concentric_shapes,
)

from pygarp.core.workers.pocket_outline import (
    calc_step_outline,
    calc_gradient_outline,
    discretize_points,
)


def generate_shape(
    shape_config: ShapeConfig,
) -> Union[Shape, ClosedSpline, Circle, Rectangle, RegularPolygon]:
    if shape_config.shape in ["shape", "spline"]:
        control_points = shape_config.control_points
        if shape_config.shape == "spline":
            shape = ClosedSpline(control_points)
        else:
            shape = Shape(control_points)
    elif shape_config.shape == "circle":
        shape = Circle(shape_config.radius, shape_config.center)
    elif shape_config.shape == "rectangle":
        shape = Rectangle(shape_config.width, shape_config.height, shape_config.center)
    elif shape_config.shape == "regular_polygon":
        shape = RegularPolygon(shape_config.side, shape_config.n, shape_config.center)
    else:
        raise ValueError()

    return shape


def generate_space(
    space_config: SpaceConfig,
) -> WorkingSpace:
    return WorkingSpace.new_space_from_three_points(
        space_config.origin, space_config.x, space_config.y
    )


def execute_pocketing_job(
    shape_config: ShapeConfig,
    space_config: SpaceConfig,
    scarfing_config: ScarfingConfig,
    robot_config: RobotConfig,
) -> npt.NDArray[np.float64]:
    shape = generate_shape(shape_config)
    space = generate_space(space_config)

    local_outline_path = None
    if scarfing_config.outline:
        if scarfing_config.outline_style == "step":
            local_outline_path = calc_step_outline(
                shape.discretize(), robot_config.gamma
            )
        else:
            local_outline_path = calc_gradient_outline(
                shape.discretize(), robot_config.gamma
            )

    fill_blocks = []
    if scarfing_config.fill:
        last_polygon = shape.polygon

        if scarfing_config.concentric:
            shapes = calc_concentric_shapes(
                shape.polygon,
                scarfing_config.c_cycle,
                scarfing_config.c_offset,
                scarfing_config.c_offset_0,
                first=False,
                flatter=True,
            )
            if shapes:
                concentric_fill_sape = calc_gradient_outline(
                    concentric_plot_fill_path(
                        shapes,
                        shape.step,
                        robot_config.exit_quote,
                        isinstance(shape, ClosedSpline),
                    ),
                    robot_config.gamma,
                )
                last_polygon = shapes[-1]
                fill_blocks.append(concentric_fill_sape)

        fill_blocks.append(_fill_pocket(scarfing_config, last_polygon, robot_config))

        if scarfing_config.recursive:
            stepped_layers = calc_concentric_shapes(
                last_polygon,
                scarfing_config.r_cycle,
                scarfing_config.r_offset,
                scarfing_config.r_offset,
                first=False,
                flatter=True,
            )

            for index, current_layer in enumerate(stepped_layers):
                profondita_z = (index + 1) * scarfing_config.z_off

                coords = np.column_stack(current_layer.exterior.xy)
                outline_grezzo = discretize_points(coords, shape.step)
                gradino_outline = (
                    calc_gradient_outline(outline_grezzo, robot_config.gamma)
                    if scarfing_config.outline_style is "gradient"
                    else calc_step_outline(outline_grezzo, robot_config.gamma)
                )
                gradino_fill = _fill_pocket(
                    scarfing_config, current_layer, robot_config
                )

                if gradino_fill.size > 0:
                    gradino_completo = np.vstack((gradino_outline, gradino_fill))
                else:
                    gradino_completo = gradino_outline

                gradino_completo[:, 2] -= profondita_z
                gradino_completo[:, 5] -= profondita_z

                fill_blocks.append(gradino_completo)

    local_fill_path = np.vstack(fill_blocks) if fill_blocks else None

    if local_outline_path is None and local_fill_path is None:
        return np.asarray([], dtype=np.float64)

    if local_outline_path is None:
        local_path = local_fill_path
    elif local_fill_path is None:
        local_path = local_outline_path
    else:
        local_path = np.vstack((local_outline_path, local_fill_path))

    return np.hstack(
        (
            space.local_to_global(local_path[:, :3]),
            space.local_to_global(local_path[:, 3:]),
        )
    )


def _fill_pocket(
    scarfing_config: ScarfingConfig, last_polygon: Polygon, robot_config: RobotConfig
):
    block = []
    if scarfing_config.fill_style == "grid":
        local_fill_path_horizontal = calc_gradient_outline(
            linear_rect_fill_path(
                calc_linear_intersection(
                    last_polygon,
                    float(scarfing_config.fill_dir % 360),
                    scarfing_config.fill_spacing,
                ),
                robot_config.exit_quote,
            ),
            robot_config.gamma,
        )
        local_fill_path_vertical = calc_gradient_outline(
            linear_rect_fill_path(
                calc_linear_intersection(
                    last_polygon,
                    float((scarfing_config.fill_dir + 90) % 360),
                    scarfing_config.fill_spacing,
                ),
                robot_config.exit_quote,
            ),
            robot_config.gamma,
        )

        local_linear_fill_path = np.vstack(
            (local_fill_path_horizontal, local_fill_path_vertical)
        )
        block.append(local_linear_fill_path)

    elif scarfing_config.fill_style == "linear":
        local_linear_fill_path = calc_gradient_outline(
            linear_rect_fill_path(
                calc_linear_intersection(
                    last_polygon,
                    float(scarfing_config.fill_dir % 360),
                    scarfing_config.fill_spacing,
                ),
                robot_config.exit_quote,
            ),
            robot_config.gamma,
        )
        block.append(local_linear_fill_path)

    return np.vstack(block) if block else np.empty((0, 3))
