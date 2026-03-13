import warnings
from typing import Union, Literal
import numpy as np
import csv
from shapely.geometry import Polygon, MultiPolygon

from src_backup.utils import *
from src_backup.obj.plane.working_plane import WorkingSpace
from src_backup.obj.robot import robot
from src_backup.obj.robot.robot import Robot
from src_backup.obj.shape.parametric_curve import ParametricCurve
from src_backup.obj.shape.circle import Circle

from src_backup.obj.shape.shape import Shape
from src_backup.obj.shape.polygon import Polygon
from src_backup.obj.shape.rectangle import Rectangle

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
            #self._ensure_pocket_closure()
            #self._ensure_pocket_plane()
            self._contour = self._calc_contour()
        return self._contour
    @property
    def fill(self) -> np.typing.NDArray[np.float64]:
        if self._fill is None:
            #self._ensure_pocket_fill()
            #self._ensure_pocket_plane()
            self._fill = self._calc_fill(7)
        return self._fill

    def _ensure_pocket_closure(self):
        x_min, x_max, y_min, y_max, _, _ = self._shape.bounding_box

        center = self._shape.barycenter
        z_center = np.zeros_like(center[0])
        center_3d = np.concatenate((center, z_center), axis=0)

        absolute_center = self._plane.local_to_global(center_3d)
        if self._robot.validate_bounding_box(bounding=(x_min, y_min, x_max, y_max), center_z=absolute_center[2]):
            raise ValueError(f"La tasca interseca l'area di sicurezza del robot")

        return
    def _ensure_pocket_plane(self):
        return # in testo momento non mi interessa fare verifiche sul piano
    def _ensure_pocket_fill(self):
        self._ensure_pocket_closure()

    def _calc_contour(self) -> np.typing.NDArray[np.float64]:
        '''head = self._shape.closure.copy()
        area = self._shape.sign_area

        if self._pocket_type == "step":
            factor = 1 if area > Eps.eps08 else -1
        elif self._pocket_type == "gradient":
            factor = 1 if area < - Eps.eps08 else -1
        else:
            raise RuntimeError(f"Pocket type {self._pocket_type} not implemented")

        distance = np.linalg.norm(head[0] - head[-1])
        closed = distance < Eps.eps08
        tail = Pocket.__calc_contour(head, factor, closed)

        z_head = np.zeros((head.shape[0], 1), dtype=head.dtype)
        z_tail = np.ones_like(z_head) * np.cos(self._robot.abs_tool_alpha)
        head_3d = np.concatenate((head, z_head), axis=1)
        tail_3d = np.concatenate((tail, z_tail), axis=1)

        absolute_head = self._plane.local_to_global(head_3d)
        absolute_tail = self._plane.local_to_global(tail_3d)

        contour = np.concatenate((absolute_head, absolute_tail), axis=1)'''
        head = self._shape.closure.copy()
        if np.allclose(head[0], head[-1], atol=1e-8):
            head = head[:-1]

        n = head.shape[0]

        prev_p = np.roll(head, 1, axis=0)  # Punto precedente
        next_p = np.roll(head, -1, axis=0)  # Punto successivo
        tangent = next_p - prev_p

        gamma = -self._robot.abs_tool_alpha
        if False:
            alpha = np.arctan2(tangent[:, 1], tangent[:, 0])
            ca, sa = np.cos(alpha), np.sin(alpha)
            cg, sg = np.cos(gamma), np.sin(gamma)

            R = np.zeros((n, 3, 3))
            R[:, 0, 0] = ca
            R[:, 1, 0] = sa
            R[:, 0, 1] = -sa * cg
            R[:, 1, 1] = ca * cg
            R[:, 2, 1] = sg
            R[:, 0, 2] = sa * sg
            R[:, 1, 2] = -ca * sg
            R[:, 2, 2] = cg

            v = R @ np.array([0, 0, 1])

            head_3d = np.column_stack((head, np.zeros(n)))
            tail = head_3d + v
            return np.hstack((head_3d, tail))

        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)

        tangent_norm[tangent_norm == 0] = 1e-10
        new_tangent = (-tangent / tangent_norm) * np.sin(gamma)
        tail_2d = head - new_tangent
        z_coord = np.cos(gamma)
        tail_z = np.full((head.shape[0], 1), z_coord)
        head_z = np.zeros((head.shape[0], 1))
        local_head = self._plane.local_to_global(np.hstack((head, head_z)))
        local_tail = self._plane.local_to_global(np.hstack((tail_2d, tail_z)))
        return np.hstack((local_head, local_tail))




    def writing(self, filename):
        closed = np.vstack((self._contour, self._contour[0, :]))
        x, y, z = closed[:, 0], closed[:, 1], closed[:, 2]
        i, j, k = closed[:, 3], closed[:, 4], closed[:, 5]


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