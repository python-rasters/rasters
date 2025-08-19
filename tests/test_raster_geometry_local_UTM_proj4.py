
import unittest
from rasters.raster_geometry import RasterGeometry
from rasters.point import Point
from rasters.CRS import CRS, WGS84

class DummyRasterGeometry(RasterGeometry):
    def __init__(self, x_min, x_max, y_min, y_max, crs=WGS84):
        super().__init__(crs=crs)
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._rows = 1
        self._cols = 1

    @property
    def x(self):
        return [[self._x_min]]

    @property
    def y(self):
        return [[self._y_min]]

    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def cell_width(self):
        return self._x_max - self._x_min

    @property
    def cell_height(self):
        return self._y_max - self._y_min

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def y_min(self):
        return self._y_min

    @property
    def y_max(self):
        return self._y_max

    @property
    def grid(self):
        return None

    @property
    def corner_polygon(self):
        return None

    @property
    def boundary(self):
        return None

    def _subset_index(self, y_slice, x_slice):
        return self

    def _slice_coords(self, y_slice, x_slice):
        return self

    def index_point(self, point):
        return (0, 0)

    def index(self, geometry):
        return (0, 0)

    def resize(self, dimensions):
        return self

    def to_dict(self):
        return {}

class TestLocalUTMProj4Property(unittest.TestCase):
    def test_local_UTM_proj4_property(self):
        # Create a dummy raster centered at lon=-120, lat=40
        raster = DummyRasterGeometry(x_min=-120.0, x_max=-120.0, y_min=40.0, y_max=40.0)
        proj4_str = raster.local_UTM_proj4
        self.assertIsInstance(proj4_str, str)
        self.assertIn("+proj=utm", proj4_str)
        self.assertIn("+zone=10", proj4_str)
        self.assertIn("+datum=WGS84", proj4_str)

if __name__ == "__main__":
    unittest.main()
