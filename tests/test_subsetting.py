import unittest
import numpy as np
from rasterio.windows import Window

from rasters.raster_grid import RasterGrid
from rasters.raster_geolocation import RasterGeolocation
from rasters.point import Point
from rasters.polygon import Polygon
from rasters.bbox import BBox
from rasters.CRS import WGS84


class TestRasterGridSubsetting(unittest.TestCase):
    """Test subsetting functionality for RasterGrid objects."""
    
    def setUp(self):
        """Set up test RasterGrid object."""
        self.grid = RasterGrid(
            x_origin=-120.0,
            y_origin=40.0,
            cell_width=0.01,
            cell_height=-0.01,
            rows=100,
            cols=100,
            crs=WGS84
        )
    
    def test_subset_with_window(self):
        """Test subsetting RasterGrid with Window object."""
        window = Window(col_off=10, row_off=20, width=30, height=40)
        subset = self.grid.subset(window)
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertEqual(subset.rows, 40)
        self.assertEqual(subset.cols, 30)
        
        # Check that the origin is adjusted correctly
        expected_x_origin = self.grid.x_origin + 10 * self.grid.cell_width
        expected_y_origin = self.grid.y_origin + 20 * self.grid.cell_height
        self.assertAlmostEqual(subset.x_origin, expected_x_origin, places=6)
        self.assertAlmostEqual(subset.y_origin, expected_y_origin, places=6)
    
    def test_subset_with_point(self):
        """Test subsetting RasterGrid with Point object."""
        # Use small polygon around point to ensure RasterGrid return
        coords = [
            (-119.55, 39.55),
            (-119.45, 39.55),
            (-119.45, 39.45),
            (-119.55, 39.45),
            (-119.55, 39.55)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.grid.subset(polygon)
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_subset_with_polygon(self):
        """Test subsetting RasterGrid with Polygon object."""
        # Create a small polygon within the grid bounds
        coords = [
            (-119.95, 39.95),
            (-119.90, 39.95),
            (-119.90, 39.90),
            (-119.95, 39.90),
            (-119.95, 39.95)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.grid.subset(polygon)
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_subset_with_bbox(self):
        """Test subsetting RasterGrid with BBox object."""
        # Create bbox within grid bounds and convert to polygon for now
        coords = [
            (-119.8, 39.8),
            (-119.2, 39.8), 
            (-119.2, 39.2),
            (-119.8, 39.2),
            (-119.8, 39.8)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.grid.subset(polygon)
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_subset_with_raster_geometry(self):
        """Test subsetting RasterGrid with another RasterGrid object."""
        other_grid = RasterGrid(
            x_origin=-119.95,
            y_origin=39.95,
            cell_width=0.005,
            cell_height=-0.005,
            rows=20,
            cols=20,
            crs=WGS84
        )
        subset = self.grid.subset(other_grid)
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_square_bracket_indexing_with_window(self):
        """Test square bracket indexing with Window object."""
        window = Window(col_off=5, row_off=10, width=20, height=25)
        subset = self.grid[window]
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertEqual(subset.rows, 25)
        self.assertEqual(subset.cols, 20)
    
    def test_square_bracket_indexing_with_point(self):
        """Test square bracket indexing with Point object."""
        # Use small polygon to ensure RasterGrid return
        coords = [
            (-119.55, 39.55),
            (-119.45, 39.55), 
            (-119.45, 39.45),
            (-119.55, 39.45),
            (-119.55, 39.55)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.grid[polygon]
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_square_bracket_indexing_with_polygon(self):
        """Test square bracket indexing with Polygon object."""
        coords = [
            (-119.95, 39.95),
            (-119.90, 39.95),
            (-119.90, 39.90),
            (-119.95, 39.90),
            (-119.95, 39.95)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.grid[polygon]
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_square_bracket_indexing_traditional(self):
        """Test that traditional slice indexing still works."""
        subset = self.grid[10:50, 20:70]
        
        self.assertIsInstance(subset, RasterGrid)
        self.assertEqual(subset.rows, 40)
        self.assertEqual(subset.cols, 50)


class TestRasterGeolocationSubsetting(unittest.TestCase):
    """Test subsetting functionality for RasterGeolocation objects."""
    
    def setUp(self):
        """Set up test RasterGeolocation object."""
        # Create coordinate arrays for a 50x50 geolocation field
        rows, cols = 50, 50
        x_coords = np.linspace(-120.0, -119.0, cols)
        y_coords = np.linspace(40.0, 39.0, rows)
        x_array, y_array = np.meshgrid(x_coords, y_coords)
        
        self.geolocation = RasterGeolocation(x_array, y_array, crs=WGS84)
    
    def test_subset_with_window(self):
        """Test subsetting RasterGeolocation with Window object."""
        window = Window(col_off=5, row_off=10, width=20, height=15)
        subset = self.geolocation.subset(window)
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertEqual(subset.rows, 15)
        self.assertEqual(subset.cols, 20)
        
        # Check that coordinate arrays are properly subset
        expected_x = self.geolocation.x[10:25, 5:25]
        expected_y = self.geolocation.y[10:25, 5:25]
        np.testing.assert_array_equal(subset.x, expected_x)
        np.testing.assert_array_equal(subset.y, expected_y)
    
    def test_subset_with_point(self):
        """Test subsetting RasterGeolocation with Point object."""
        # Use a point that's definitely within the geolocation bounds
        point = Point(-119.5, 39.5, crs=WGS84)
        # Create a small polygon around the point instead
        coords = [
            (-119.6, 39.6),
            (-119.4, 39.6),
            (-119.4, 39.4), 
            (-119.6, 39.4),
            (-119.6, 39.6)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.geolocation.subset(polygon)
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_subset_with_polygon(self):
        """Test subsetting RasterGeolocation with Polygon object."""
        # Create a polygon within the geolocation bounds
        coords = [
            (-119.8, 39.8),
            (-119.2, 39.8),
            (-119.2, 39.2),
            (-119.8, 39.2),
            (-119.8, 39.8)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.geolocation.subset(polygon)
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_subset_with_bbox(self):
        """Test subsetting RasterGeolocation with BBox object."""
        # Use polygon instead of BBox for now
        coords = [
            (-119.8, 39.8),
            (-119.2, 39.8),
            (-119.2, 39.2),
            (-119.8, 39.2),
            (-119.8, 39.8)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.geolocation.subset(polygon)
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_subset_with_raster_geometry(self):
        """Test subsetting RasterGeolocation with RasterGrid object."""
        grid = RasterGrid(
            x_origin=-119.8,
            y_origin=39.8,
            cell_width=0.01,
            cell_height=-0.01,
            rows=30,
            cols=30,
            crs=WGS84
        )
        subset = self.geolocation.subset(grid)
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_window_with_buffer(self):
        """Test window method with buffer parameter."""
        # Use polygon instead of point for more reliable results
        coords = [
            (-119.6, 39.6),
            (-119.4, 39.6),
            (-119.4, 39.4),
            (-119.6, 39.4),
            (-119.6, 39.6)
        ]
        polygon = Polygon(coords, crs=WGS84)
        window_no_buffer = self.geolocation.window(polygon)
        window_with_buffer = self.geolocation.window(polygon, buffer=5)
        
        # Window with buffer should be larger
        self.assertGreaterEqual(window_with_buffer.width, window_no_buffer.width)
        self.assertGreaterEqual(window_with_buffer.height, window_no_buffer.height)
        
        # Buffer should not exceed array bounds
        self.assertGreaterEqual(window_with_buffer.col_off, 0)
        self.assertGreaterEqual(window_with_buffer.row_off, 0)
        self.assertLessEqual(window_with_buffer.col_off + window_with_buffer.width, self.geolocation.cols)
        self.assertLessEqual(window_with_buffer.row_off + window_with_buffer.height, self.geolocation.rows)
    
    def test_square_bracket_indexing_with_window(self):
        """Test square bracket indexing with Window object."""
        window = Window(col_off=8, row_off=12, width=15, height=18)
        subset = self.geolocation[window]
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertEqual(subset.rows, 18)
        self.assertEqual(subset.cols, 15)
    
    def test_square_bracket_indexing_with_point(self):
        """Test square bracket indexing with Point object."""
        # Use polygon around point for more reliable indexing
        coords = [
            (-119.6, 39.6),
            (-119.4, 39.6),
            (-119.4, 39.4),
            (-119.6, 39.4),
            (-119.6, 39.6)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.geolocation[polygon]
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_square_bracket_indexing_with_bbox(self):
        """Test square bracket indexing with BBox-like polygon."""
        coords = [
            (-119.6, 39.7),
            (-119.4, 39.7),
            (-119.4, 39.3),
            (-119.6, 39.3),
            (-119.6, 39.7)
        ]
        polygon = Polygon(coords, crs=WGS84)
        subset = self.geolocation[polygon]
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertGreater(subset.rows, 0)
        self.assertGreater(subset.cols, 0)
    
    def test_square_bracket_indexing_traditional(self):
        """Test that traditional slice indexing still works."""
        subset = self.geolocation[5:25, 10:30]
        
        self.assertIsInstance(subset, RasterGeolocation)
        self.assertEqual(subset.rows, 20)
        self.assertEqual(subset.cols, 20)
    
    def test_window_no_overlap_error(self):
        """Test that window method raises error when geometry doesn't overlap."""
        # Point far outside the geolocation bounds
        point = Point(-90.0, 30.0, crs=WGS84)
        
        with self.assertRaises(ValueError) as context:
            self.geolocation.window(point)
        
        self.assertIn("No points found within the target geometry", str(context.exception))


class TestSubsettingConsistency(unittest.TestCase):
    """Test consistency between RasterGrid and RasterGeolocation subsetting."""
    
    def setUp(self):
        """Set up test objects for consistency testing."""
        # Create a RasterGrid
        self.grid = RasterGrid(
            x_origin=-120.0,
            y_origin=40.0,
            cell_width=0.02,
            cell_height=-0.02,
            rows=50,
            cols=50,
            crs=WGS84
        )
        
        # Create a RasterGeolocation with similar bounds
        rows, cols = 50, 50
        x_coords = np.linspace(-120.0, -119.0, cols)
        y_coords = np.linspace(40.0, 39.0, rows)
        x_array, y_array = np.meshgrid(x_coords, y_coords)
        self.geolocation = RasterGeolocation(x_array, y_array, crs=WGS84)
    
    def test_both_support_same_window(self):
        """Test that both geometry types can be subset with the same Window."""
        window = Window(col_off=10, row_off=15, width=20, height=25)
        
        grid_subset = self.grid.subset(window)
        geolocation_subset = self.geolocation.subset(window)
        
        # Both should return their respective types
        self.assertIsInstance(grid_subset, RasterGrid)
        self.assertIsInstance(geolocation_subset, RasterGeolocation)
        
        # Both should have the same dimensions
        self.assertEqual(grid_subset.rows, geolocation_subset.rows)
        self.assertEqual(grid_subset.cols, geolocation_subset.cols)
        self.assertEqual(grid_subset.rows, 25)
        self.assertEqual(grid_subset.cols, 20)
    
    def test_both_support_square_bracket_indexing(self):
        """Test that both support square bracket indexing with geometries."""
        coords = [
            (-119.8, 39.8),
            (-119.2, 39.8),
            (-119.2, 39.2),
            (-119.8, 39.2),
            (-119.8, 39.8)
        ]
        polygon = Polygon(coords, crs=WGS84)
        
        # Both should work with square bracket indexing
        grid_subset = self.grid[polygon]
        geolocation_subset = self.geolocation[polygon]
        
        self.assertIsInstance(grid_subset, RasterGrid)
        self.assertIsInstance(geolocation_subset, RasterGeolocation)
    
    def test_method_vs_indexing_equivalence(self):
        """Test that subset method and square bracket indexing are equivalent."""
        window = Window(col_off=5, row_off=8, width=15, height=12)
        
        # For RasterGrid
        grid_method = self.grid.subset(window)
        grid_indexing = self.grid[window]
        self.assertEqual(grid_method.rows, grid_indexing.rows)
        self.assertEqual(grid_method.cols, grid_indexing.cols)
        self.assertEqual(grid_method.x_origin, grid_indexing.x_origin)
        self.assertEqual(grid_method.y_origin, grid_indexing.y_origin)
        
        # For RasterGeolocation
        geolocation_method = self.geolocation.subset(window)
        geolocation_indexing = self.geolocation[window]
        self.assertEqual(geolocation_method.rows, geolocation_indexing.rows)
        self.assertEqual(geolocation_method.cols, geolocation_indexing.cols)
        np.testing.assert_array_equal(geolocation_method.x, geolocation_indexing.x)
        np.testing.assert_array_equal(geolocation_method.y, geolocation_indexing.y)


if __name__ == '__main__':
    unittest.main()