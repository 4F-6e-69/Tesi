from src.obj.robot import robot
from src.obj.robot.robot import Robot

class Pocket(object):
    def __init__(self, robot_model: Robot):
        self._robot = robot_model

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