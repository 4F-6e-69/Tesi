import numpy as np

from pygarp.core.models.commons import Eps, Shapes, TransformationRef
from pygarp.core.models.validators import ShapeConfig, TransformConfig
from pygarp.core.models.shapes.shape.core import Shape
from pygarp.core.models.shapes.circle import Circle
from pygarp.core.models.shapes.rectangle import Rectangle
from pygarp.core.models.shapes.regular_polygon import RegularPolygon
from pygarp.core.models.shapes.closed_spline import ClosedSpline


def generate_shape(
    shape_config: ShapeConfig,
) -> Shapes:
    origin = (
        np.zeros(2, dtype=np.float64)
        if shape_config.origin is None
        else shape_config.origin
    )
    eps = Eps.eps010 if shape_config.eps is None else shape_config.eps

    if shape_config.shape in ["shape", "spline"]:
        control_points = shape_config.control_points
        if shape_config.shape == "spline":
            shape = ClosedSpline(
                control_points,
                origin=origin,
                eps=eps,
            )
        else:
            shape = Shape(
                control_points,
                origin=origin,
                assume_sort=shape_config.assume_sort,
                eps=eps,
            )
    elif shape_config.shape == "circle":
        shape = Circle(
            shape_config.radius,
            shape_config.center,
            origin=origin,
            eps=eps,
        )
    elif shape_config.shape == "rectangle":
        shape = Rectangle(
            shape_config.width,
            shape_config.height,
            shape_config.center,
            origin=origin,
            eps=eps,
        )
    elif shape_config.shape == "regular_polygon":
        shape = RegularPolygon(
            shape_config.side,
            shape_config.n,
            shape_config.center,
            origin=origin,
            eps=eps,
        )
    else:
        raise ValueError()

    return shape


def transform_shape(transform_config: TransformConfig, shape: Shapes) -> Shapes:
    if transform_config.translation is not None:
        shape.translate(
            transform_config.translation[0], transform_config.translation[1]
        )

    ref_point: TransformationRef = (
        "origin" if transform_config.ref is None else transform_config.ref
    )
    if transform_config.rotation is not None:
        shape.rotate(transform_config.rotation, ref_point)

    if transform_config.scale is not None:
        shape.scale(transform_config.scale[0], transform_config.scale[1], ref_point)

    return shape
