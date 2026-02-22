import unittest

import numpy as np

from src.obj.robot.robot import Robot


class TestRobot(unittest.TestCase):
    def setUp(self):
        self.robot_init = Robot()
        self.robot_init.max_radius = 100
        self.robot_init.min_radius = 50
        self.robot_init.abs_tool_radius = 5
        self.robot_init.abs_tool_alpha = 7

        self.equal = True
        self.robot_constructor = Robot.new_robot(100, 50, 7, 5)

        self.robot_empty = Robot()

    def test_constructor(self):
        assert isinstance(self.robot_init, Robot)

        assert isinstance(self.robot_init.max_radius, float)
        assert isinstance(self.robot_init.min_radius, float)
        assert isinstance(self.robot_init.abs_tool_radius, float)
        assert isinstance(self.robot_init.abs_tool_alpha, float)

        assert isinstance(self.robot_constructor, Robot)

        assert isinstance(self.robot_init.max_radius, float)
        assert isinstance(self.robot_init.min_radius, float)
        assert isinstance(self.robot_init.abs_tool_radius, float)
        assert isinstance(self.robot_init.abs_tool_alpha, float)

        assert isinstance(self.robot_empty, Robot)

        if self.equal:
            init_data = np.asarray([self.robot_init.max_radius, self.robot_init.min_radius, self.robot_init.abs_tool_radius, self.robot_init.abs_tool_alpha])
            constructor_data = np.asarray([self.robot_constructor.max_radius, self.robot_constructor.min_radius, self.robot_constructor.abs_tool_radius, self.robot_constructor.abs_tool_alpha])

            assert np.allclose(init_data, constructor_data, np.finfo(np.float64).eps)
    def test_value(self):
        with self.assertRaisesRegex(AttributeError, 'is not set'):
            _ = self.robot_empty.max_radius
        with self.assertRaisesRegex(AttributeError, 'is not set'):
            _ = self.robot_empty.min_radius
        with self.assertRaisesRegex(AttributeError, 'is not set'):
            _ = self.robot_empty.abs_tool_radius
        with self.assertRaisesRegex(AttributeError, 'is not set'):
            _ = self.robot_empty.abs_tool_alpha

        with self.assertRaisesRegex(ValueError, 'must be'):
            self.robot_init.max_radius = 40
        with self.assertRaisesRegex(ValueError, 'must be'):
            self.robot_init.min_radius = 120
    def test_validation_points(self):
        test_cases = [
            ([100, 120], False),  # Fuori raggio (norma ~156)
            ([80, 40], True),  # Valido (norma ~89.4) -> Ho corretto il punto!
            ([0, 2], False),  # Sotto il minimo (norma 2)
            ([0, 51], True),  # Valido (norma 51)
        ]

        for point, expected_result in test_cases:
            with self.subTest(point=point):
                result = self.robot_init.validate_point(point)

                self.assertIsInstance(result, bool)
                self.assertEqual(result, expected_result)
    def test_validate_bounding_box(self):
        self.robot_init.max_radius = 100.0
        self.robot_init.min_radius = 50.0

        test_cases_bounding = [
            ((10, 0, 10, 0), [75, 0], True, "Valid bounding box inside workspace"),
            ((5, 5, 5, 5), [30, 0], False, "Center is outside workspace (too close)"),
            ((10, 0, 20, 0), [90, 0], False, "p_max exceeds max_radius"),
            ((20, 0, 10, 0), [60, 0], False, "p_min exceeds min_radius"),
            ((10, 10, 10, 10), [60, 40], True, "Valid diagonal bounding box"),
        ]

        for bounding, center, expected_result, description in test_cases_bounding:
            with self.subTest(msg=description, bounding=bounding, center=center):
                result = self.robot_init.validate_bounding_box(bounding, center)

                self.assertIsInstance(result, bool)
                self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()