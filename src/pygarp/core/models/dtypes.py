from typing import Union

from pygarp.core.models.shapes.circle import Circle
from pygarp.core.models.shapes.rectangle import Rectangle
from pygarp.core.models.shapes.regular_polygon import RegularPolygon
from pygarp.core.models.shapes.closed_spline import ClosedSpline
from pygarp.core.models.shapes.shape.core import Shape

Shapes = Union[Shape, ClosedSpline, Circle, Rectangle, RegularPolygon]
