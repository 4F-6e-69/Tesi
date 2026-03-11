import unittest
import numpy as np
from shapely.geometry import Polygon

from src.models.shape.shape import Shape

class TestShape(unittest.TestCase):

    def setUp(self):
        self.square_points = [[0, 0], [2, 0], [2, 2], [0, 2]]
        self.default_shape = Shape(points=self.square_points, origin=None)
        self.shape = Shape(points=self.square_points, origin=[0, 0])

    def test_init_with_none_origin(self):
        shape = Shape(points=self.square_points, origin=None)

        self.assertIsInstance(shape.shapely, Polygon)
        np.testing.assert_array_equal(shape.origin, np.zeros(2))

        self.assertFalse(shape.origin_is_center)
    def test_init_with_custom_origin(self):
        custom_origin = [1, 1]
        shape = Shape(points=self.square_points, origin=custom_origin)

        np.testing.assert_array_equal(shape.origin, np.array([1.0, 1.0]))
        self.assertTrue(shape.origin_is_center)
    def test_area_calculation_and_cache(self):
        self.assertIsNone(self.default_shape._area)
        self.assertEqual(self.default_shape.area, 4.0)
        self.assertEqual(self.default_shape._area, 4.0)  # Verifica che sia stato cachato
    def test_length_calculation_and_cache(self):
        self.assertIsNone(self.default_shape._length)
        self.assertEqual(self.default_shape.length, 8.0)
        self.assertEqual(self.default_shape._length, 8.0)  # Verifica cache
    def test_bounds_calculation_and_cache(self):
        self.assertIsNone(self.default_shape._bounds)
        self.assertEqual(self.default_shape.bounds, (0.0, 0.0, 2.0, 2.0))
        self.assertEqual(self.default_shape._bounds, (0.0, 0.0, 2.0, 2.0))
    def test_barycenter_calculation_and_cache(self):
        self.assertIsNone(self.default_shape._barycenter)
        np.testing.assert_array_equal(self.default_shape.barycenter, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(self.default_shape._barycenter, np.array([1.0, 1.0]))
    def test_origin_setter(self):
        self.assertFalse(self.default_shape.origin_is_center)

        self.default_shape.origin = [1.0, 1.0]
        np.testing.assert_array_equal(self.default_shape.origin, np.array([1.0, 1.0]))
        self.assertTrue(self.default_shape.origin_is_center)
    def test_reset_cache_all(self):
        _ = self.default_shape.area
        _ = self.default_shape.length
        _ = self.default_shape.bounds
        _ = self.default_shape.barycenter

        self.default_shape.reset_cache()

        self.assertIsNone(self.default_shape._area)
        self.assertIsNone(self.default_shape._length)
        self.assertIsNone(self.default_shape._bounds)
        self.assertIsNone(self.default_shape._barycenter)
    def test_reset_with_resets_iterable(self):
        _ = self.default_shape.area
        _ = self.default_shape.length

        self.default_shape.reset(resets=["area"])

        self.assertIsNone(self.default_shape._area)
        self.assertEqual(self.default_shape._length, 8.0)  # Il perimetro dovrebbe ancora essere in cache
    def test_reset_with_none(self):
        _ = self.default_shape.area
        self.default_shape.reset(resets=None)
        self.assertIsNone(self.default_shape._area)
    def test_str_method(self):
        self.assertEqual(str(self.default_shape), "POLYGON")
    def test_repr_method(self):
        self.assertIn("POLYGON", repr(self.default_shape))
    def test_shapely_property_warning(self):
        self.default_shape._shapely_shape = None

        with self.assertWarnsRegex(UserWarning, "L'oggetto Shapely non è stato ancora inizializzato"):
            shape_obj = self.default_shape.shapely
            self.assertIsNone(shape_obj)

    def test_translate_updates_geometry_and_cache(self):
        original_area = self.shape.area
        original_length = self.shape.length

        result = self.shape.translate(x_off=1.0, y_off=2.0)

        self.assertIs(result, self.shape)

        self.assertEqual(self.shape.bounds, (1.0, 2.0, 3.0, 4.0))
        np.testing.assert_array_equal(self.shape.barycenter, np.array([2.0, 3.0]))

        np.testing.assert_array_equal(self.shape.origin, np.array([1.0, 2.0]))

        self.assertIsNotNone(self.shape._area)
        self.assertIsNotNone(self.shape._length)
        self.assertEqual(self.shape.area, original_area)
    def test_rotate_around_barycenter(self):
        original_bounds = self.shape.bounds

        self.shape.rotate(angle=90.0, ref="barycenter")
        new_bounds = self.shape.bounds
        for orig, new in zip(original_bounds, new_bounds):
            self.assertAlmostEqual(orig, new, places=5)
        self.assertEqual(self.shape.area, 4.0)
    def test_rotate_around_origin(self):
        self.shape.rotate(angle=180.0, ref="origin")

        minx, miny, maxx, maxy = self.shape.bounds
        self.assertAlmostEqual(minx, -2.0, places=5)
        self.assertAlmostEqual(miny, -2.0, places=5)
        self.assertAlmostEqual(maxx, 0.0, places=5)
        self.assertAlmostEqual(maxy, 0.0, places=5)
    def test_scale_resets_entire_cache(self):
        _ = self.shape.area
        _ = self.shape.length
        _ = self.shape.bounds
        _ = self.shape.barycenter

        self.shape.scale(x_fact=2.0, y_fact=2.0, ref="origin")

        self.assertIsNone(self.shape._area)
        self.assertIsNone(self.shape._length)
        self.assertIsNone(self.shape._bounds)
        self.assertIsNone(self.shape._barycenter)

        self.assertEqual(self.shape.area, 16.0)
        self.assertEqual(self.shape.length, 16.0)
        self.assertEqual(self.shape.bounds, (0.0, 0.0, 4.0, 4.0))
    def test_transformations_method_chaining(self):
        self.shape.translate(x_off=2.0, y_off=0.0) \
            .scale(x_fact=2.0, y_fact=2.0, ref="origin") \
            .rotate(angle=90.0, ref="barycenter")

        self.assertIsInstance(self.shape.shapely, Polygon)


if __name__ == '__main__':
    unittest.main()