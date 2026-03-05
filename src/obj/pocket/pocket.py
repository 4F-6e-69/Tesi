import warnings
from typing import Union, Literal
import numpy as np
import csv

from src.obj.plane.working_plane import WorkingSpace
from src.obj.robot import robot
from src.obj.robot.robot import Robot
from src.obj.shape.parametric_curve import ParametricCurve
from src.obj.shape.circle import Circle

from src.obj.shape.shape import Shape
from src.obj.shape.polygon import Polygon
from src.obj.shape.rectangle import Rectangle

Shapes = Union[Shape, Polygon, Rectangle, ParametricCurve, Circle, None]
PocketType = Literal["step", "gradient"]

class Pocket(object):
    def __init__(self, robot_model: Robot, shape: Shapes, plane: WorkingSpace, pocket_type: PocketType = "step"):
        self._robot: Robot = robot_model
        self._shape: Shapes = shape
        self._plane:WorkingSpace = plane

        self._contour: np.typing.NDArray[np.float64] | None = None
        self._fill: np.typing.NDArray[np.float64] | None = None

        if pocket_type != "step" and pocket_type != "gradient":
            warnings.warn("PocketType non valido -> casting implicito a 'step'")
            self._pocket_type: PocketType = "step"

        else:
            self._pocket_type: PocketType = pocket_type

    @property
    def robot(self) -> Robot:
        if robot is None:
            raise AttributeError("robot for the pocket is not set")
        return self._robot
    @robot.setter
    def robot(self, new_robot_model: Robot):
        if new_robot_model is None and not isinstance(new_robot_model, Robot) and  not Robot.validate_model(new_robot_model):
            raise AttributeError("ne robot for the pocket is not valid")

        self._robot = new_robot_model

    @property
    def shape(self) -> Shapes:
        if self._shape is None:
            raise AttributeError("shape for the pocket is not set")
        return self._shape
    @shape.setter
    def shape(self, new_shape: Shapes):
        if not isinstance(new_shape, Shapes):
            raise TypeError("shape non valida")

        self._shape = new_shape


    @property
    def contour(self) -> np.typing.NDArray[np.float64]:
        if self._contour is None:
            self._ensure_pocket_closure()
            self._ensure_pocket_plane()
            self._contour = self._calc_contour()
        return self._contour
    @property
    def fill(self) -> np.typing.NDArray[np.float64]:
        if self._fill is None:
            self._ensure_pocket_fill()
            self._ensure_pocket_plane()
            self._fill = self._calc_fill()
        return self._fill

    def _ensure_pocket_closure(self):
        x_min, x_max, y_min, y_max = self._shape.boundary

        center = self._shape.center
        z_center = np.ones((center.shape[0], 1), dtype=center.dtype)
        center_3d = np.concatenate((center, z_center), axis=1)

        absolute_center = self._plane.local_to_global(center_3d)
        if self._robot.validate_bounding_box(bounding=(x_min, y_min, x_max, y_max), center_z=absolute_center[2]):
            raise ValueError(f"La tasca interseca l'area di sicurezza del robot")

        return
    def _ensure_pocket_plane(self):
        return # in testo momento non mi interessa fare verifiche sul piano
    def _ensure_pocket_fill(self):
        self._ensure_pocket_closure()

    def _calc_contour(self) -> np.typing.NDArray[np.float64]:
        head = self._shape.closure.copy()
        area = self._shape.sign_area

        if self._pocket_type == "step":
            factor = 1 if area > WorkingSpace.EPS_12 else -1
        elif self._pocket_type == "gradient":
            factor = 1 if area < - WorkingSpace.EPS_12 else -1
        else:
            raise RuntimeError(f"Pocket type {self._pocket_type} not implemented")

        distance = np.linalg.norm(head[0] - head[-1])
        closed = distance < WorkingSpace.EPS_12
        tail = Pocket.__calc_contour(head, factor, closed)

        z_head = np.zeros((head.shape[0], 1), dtype=head.dtype)
        z_tail = np.ones_like(z_head) * np.cos(self._robot.abs_tool_alpha)
        head_3d = np.concatenate((head, z_head), axis=1)
        tail_3d = np.concatenate((tail, z_tail), axis=1)

        absolute_head = self._plane.local_to_global(head_3d)
        absolute_tail = self._plane.local_to_global(tail_3d)

        contour = np.concatenate((absolute_head, absolute_tail), axis=1)
        return np.concatenate((contour, contour[0:1]), axis=0)
    @staticmethod
    def __calc_contour(points: np.typing.NDArray[np.float64], epsilon: float = WorkingSpace.EPS_12, factor: int = 1, closed: bool = False) -> np.typing.NDArray[np.float64]:
        if closed:
            points = np.delete(points, [len(points) - 1], axis=0)

        next_points = np.roll(points, shift=-1, axis=0)
        previous_points = np.roll(points, shift=1, axis=0)
        tangent = next_points - previous_points

        if factor > 0:
            factor = float(1.0)
        else:
            factor = float(-1.0)

        normal = np.zeros_like(tangent)
        normal[:, 0] = -factor * tangent[:, 1]
        normal[:, 1] = factor * tangent[:, 0]

        norm_magnitude = np.linalg.norm(normal, axis=1, keepdims=True)
        normal_unit = normal / (norm_magnitude + epsilon)

        tail = points + normal_unit
        return tail

    # TODO
    def _calc_fill(self) -> np.typing.NDArray[np.float64]:
        pass

    def writing(self, filename):
        x, y, z = self._contour[:, 0], self._contour[:, 1], self._contour[:, 2]
        i, j, k = self._contour[:, 3], self._contour[:, 4], self._contour[:, 5]

        A = np.zeros((len(x), 3))
        for o in range(len(x)):
            T = np.array([[1, 0, 0, -x[o]],
                          [0, 1, 0, -y[o]],
                          [0, 0, 1, -z[o]],
                          [0, 0, 0, 1]])

            vet = np.array([[i[o]], [j[o]], [k[o]], [1]])
            temp = np.dot(T, vet)
            A[o, :3] = temp[:3].flatten()

        A = np.column_stack((x, y, z, -A))
        A = A.T
        A = A.tolist()

        with open(filename, mode='w', newline='') as file_csv:
            writer = csv.writer(file_csv)
            for riga in range(len(x)):
                writer.writerow([
                    "{:.3f}".format(A[0][riga]),
                    " {:.3f}".format(A[1][riga]),
                    " {:.3f}".format(A[2][riga]),
                    " {:.3f}".format(A[3][riga]),
                    " {:.3f}".format(A[4][riga]),
                    " {:.3f}".format(A[5][riga])
                ])