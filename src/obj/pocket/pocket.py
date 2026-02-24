from typing import Union

from src.obj.robot import robot
from src.obj.robot.robot import Robot

from src.obj.shape.shape import Shape
from src.obj.shape.polygon import Polygon
from src.obj.shape.rectangle import Rectangle

Shapes = Union[Shape, Polygon, Rectangle, None]
class Pocket(object):
    def __init__(self, robot_model: Robot):
        self._robot = robot_model
        self._shape = Shapes

    @property
    def robot(self):
        if robot is None:
            raise AttributeError("robot for the pocket is not set")
        return self._robot
    @robot.setter
    def robot(self, new_robot_model: Robot):
        if new_robot_model is None and not isinstance(new_robot_model, Robot) and  not Robot.validate_model(new_robot_model):
            raise AttributeError("ne robot for the pocket is not valid")

        self._robot = new_robot_model

    @property
    def shape(self):
        if self._shape is None:
            raise AttributeError("shape for the pocket is not set")
        return self._shape
    @shape.setter
    def shape(self, new_shape: Shapes):
        if not isinstance(new_shape, Shapes):
            raise TypeError("shape non valida")

        self._shape = new_shape