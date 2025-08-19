import unittest
import warnings
from rasters.local_UTM_proj4 import local_UTM_proj4
from rasters.point import Point
from pyproj import CRS

class TestLocalUTMProj4(unittest.TestCase):
    def test_utm_zone_integer(self):
        # Test for a point in UTM zone 10
        point = Point(-120.0, 40.0)  # lon, lat
        crs = local_UTM_proj4(point)
        self.assertIsInstance(crs, CRS)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You will likely lose important projection information when converting to a PROJ string from another format.",
                category=UserWarning,
                module="pyproj"
            )
            proj_str = crs.to_proj4()
        self.assertIn("+zone=10 ", proj_str)
        self.assertIn("+proj=utm", proj_str)
        self.assertIn("+datum=WGS84", proj_str)

    def test_utm_zone_southern_hemisphere(self):
        # Test for a point in southern hemisphere
        point = Point(30.0, -20.0)  # lon, lat
        crs = local_UTM_proj4(point)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You will likely lose important projection information when converting to a PROJ string from another format.",
                category=UserWarning,
                module="pyproj"
            )
            proj_str = crs.to_proj4()
        self.assertIn("+south", proj_str)

    def test_invalid_longitude(self):
        # Test for invalid longitude
        point = Point(200.0, 40.0)
        with self.assertRaises(ValueError):
            local_UTM_proj4(point)

if __name__ == "__main__":
    unittest.main()
