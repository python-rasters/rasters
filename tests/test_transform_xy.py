import numpy as np
from pyproj import CRS
from rasters.transform_xy import transform_xy

def test_transform_xy_geographic_to_projected():
    # WGS84 geographic CRS
    source_crs = CRS.from_epsg(4326)
    # UTM zone 33N projected CRS
    target_crs = CRS.from_epsg(32633)
    # Example coordinates: longitude, latitude
    x = np.array([12.0, 13.0])
    y = np.array([55.0, 56.0])
    x_t, y_t = transform_xy(x, y, source_crs, target_crs)
    # Check output shape and type
    assert x_t.shape == x.shape
    assert y_t.shape == y.shape
    assert np.all(np.isfinite(x_t))
    assert np.all(np.isfinite(y_t))
    # Check that transformed coordinates are not equal to input
    assert not np.allclose(x, x_t)
    assert not np.allclose(y, y_t)

def test_transform_xy_projected_to_geographic():
    # UTM zone 33N projected CRS
    source_crs = CRS.from_epsg(32633)
    # WGS84 geographic CRS
    target_crs = CRS.from_epsg(4326)
    # Example coordinates: easting, northing
    x = np.array([500000, 600000])
    y = np.array([6100000, 6200000])
    x_t, y_t = transform_xy(x, y, source_crs, target_crs)
    # Check output shape and type
    assert x_t.shape == x.shape
    assert y_t.shape == y.shape
    assert np.all(np.isfinite(x_t))
    assert np.all(np.isfinite(y_t))
    # Check that transformed coordinates are not equal to input
    assert not np.allclose(x, x_t)
    assert not np.allclose(y, y_t)

def test_transform_xy_clip_geographic():
    # WGS84 geographic CRS
    source_crs = CRS.from_epsg(4326)
    target_crs = CRS.from_epsg(4326)
    # Coordinates outside valid bounds
    x = np.array([-200, 0, 200])
    y = np.array([-100, 0, 100])
    x_t, y_t = transform_xy(x, y, source_crs, target_crs)
    # Out-of-bounds should be nan
    assert np.isnan(x_t[0])
    assert np.isnan(x_t[2])
    assert np.isnan(y_t[0])
    assert np.isnan(y_t[2])
    # Valid should remain
    assert x_t[1] == 0
    assert y_t[1] == 0
