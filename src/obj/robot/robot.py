import numpy as np

class Robot(object):
    def __init__(self):
        # Raggio di azione del robot
        self._max_radius: float | None = None
        self._min_radius: float | None = None

        # Caratteristi dell'utensile usato
        self._abs_tool_alpha: float | None = None
        self._abs_tool_radius: float | None = None

        # Tolleranza per i calcoli
        self._tolerance = np.finfo(np.float64).eps

    @classmethod
    def new_robot(cls, max_radius=None, min_radius=None, abs_tool_alpha=None, abs_tool_radius=None):
        robot = cls()

        robot.max_radius = max_radius
        robot.min_radius = min_radius

        robot.abs_tool_alpha = abs_tool_alpha
        robot.abs_tool_radius = abs_tool_radius

        return robot

    @property
    def max_radius(self) -> float:
        if self._max_radius is None:
            raise AttributeError('max_radius is not set')
        return self._max_radius
    @property
    def min_radius(self) -> float:
        if self._min_radius is None:
            raise AttributeError('min_radius is not set')
        return self._min_radius

    @max_radius.setter
    def max_radius(self, value: float):
        if self._min_radius is not None and value < self.min_radius:
            raise ValueError('max_radius must be greater than min_radius')

        self._max_radius = np.abs(value, dtype=np.float64)
    @min_radius.setter
    def min_radius(self, value: float):
        if self._max_radius is not None and value > self.max_radius:
            raise ValueError('min_radius must be less than max_radius')

        self._min_radius = np.abs(value, dtype=np.float64)

    @property
    def abs_tool_alpha(self) -> float:
        if self._abs_tool_alpha is None:
            raise AttributeError('abs_tool_alpha is not set')

        return self._abs_tool_alpha
    @abs_tool_alpha.setter
    def abs_tool_alpha(self, value: float):
        self._abs_tool_alpha = np.abs(value, dtype=np.float64)

    @property
    def abs_tool_radius(self) -> float:
        if self._abs_tool_radius is None:
            raise AttributeError('abs_tool_radius is not set')

        return self._abs_tool_radius
    @abs_tool_radius.setter
    def abs_tool_radius(self, value: float):
        self._abs_tool_radius = np.abs(value, dtype=np.float64)

    @property
    def tolerance(self):
        return self._tolerance
    @tolerance.setter
    def tolerance(self, value: float):
        self._tolerance = np.abs(value, dtype=np.float64)

    def validate_point(self, point) -> bool: # TODO definire un tipe hint per le coordinate a 3 dimensioni
        p = np.asarray(point)
        p_norm = np.linalg.norm(p)

        if self._min_radius < p_norm < self.max_radius:
            return True

        return False
    def validate_bounding_box(self, bounding: tuple[float, float, float, float], center) -> bool: # TODO definire un tipe hint per le coordinate a 3 dimensioni
        c = np.asarray(center)
        if not self.validate_point(c):
            return False

        min_x, min_y, max_x, max_y = bounding

        p_min = c - np.asarray([min_x, min_y], dtype=np.float64)
        p_min_norm = np.linalg.norm(p_min)

        p_max = c + np.asarray([max_x, max_y], dtype=np.float64)
        p_max_norm = np.linalg.norm(p_max)

        if (self.min_radius < p_min_norm < self.max_radius) and (self.min_radius < p_max_norm < self.max_radius):
            return True

        return False
    def validate_shape_angles(self, angels: tuple[float, float]):
        print(f"angels: {angels}, robot tool width: {self.abs_tool_radius}, robot tool angle: {self.abs_tool_alpha}")
        return NotImplemented

    def is_valid_model(self) -> bool:
        if (self.min_radius is None and
                self.max_radius is None and
                self.tolerance is None and
                self.abs_tool_radius > self.tolerance and
                self.abs_tool_alpha > self.tolerance and
                self.min_radius < self.max_radius):
            return False
        return True
    @staticmethod
    def validate_model(robot_instance):
        if not isinstance(robot_instance, Robot):
            raise AttributeError("robot_instance is not an instance of Robot")

        return robot_instance.is_valid_model()