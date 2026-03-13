import numpy as np

from src.models.shape.generic_shape import Shape

class GenericPolygon(Shape):

    @property
    def open_vertices(self):
        x, y = self.shapely.exterior.coords.xy
        return np.delete(np.column_stack((x, y)), -1)
    @property
    def closed_vertices(self):
        x, y = self.shapely.exterior.coords
        return np.column_stack((x, y))