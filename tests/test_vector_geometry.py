import warnings
import numpy as np
from rasters import Point, MultiPoint, Polygon, Raster, RasterGrid, WGS84


def _assert_no_deprecations(caught):
    deprecations = [
        w for w in caught
        if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
    ]
    assert len(deprecations) == 0, [str(w.message) for w in deprecations]


def test_point_to_crs_no_deprecation_warning():
    point = Point(-118.5, 34.5, crs=WGS84)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = point.to_crs("EPSG:32611")

    _assert_no_deprecations(caught)
    assert projected.crs.to_epsg() == 32611


def test_point_3d_to_crs_preserves_z():
    point = Point(-118.5, 34.5, 100.0, crs=WGS84)
    projected = point.to_crs("EPSG:32611")

    assert projected.geometry.has_z
    assert np.isclose(projected.geometry.z, 100.0)


def test_polygon_to_crs_no_deprecation_warning():
    polygon = Polygon(
        [(-119.0, 34.0), (-118.0, 34.0), (-118.0, 35.0), (-119.0, 35.0), (-119.0, 34.0)],
        crs=WGS84,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = polygon.to_crs("EPSG:32611")

    _assert_no_deprecations(caught)
    assert projected.crs.to_epsg() == 32611


def test_empty_multipoint_to_crs_no_deprecation_warning():
    mp = MultiPoint(crs=WGS84)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = mp.to_crs("EPSG:32611")

    _assert_no_deprecations(caught)
    assert projected.geometry.is_empty
    assert projected.crs.to_epsg() == 32611


def test_raster_grid_operations_no_deprecation_warning():
    grid = RasterGrid.from_bbox(
        xmin=-119.0, ymin=34.0, xmax=-118.0, ymax=35.0, cell_size=0.1, crs=WGS84
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crosses = grid.crosses_antimeridian
        _ = grid.xy
        _ = grid.x_vector
        _ = grid.y_vector
        _ = grid.boundary
        _ = grid.index_point(Point(-118.5, 34.5, crs=WGS84))
        _ = grid.shift_xy(0.1, -0.1)

    _assert_no_deprecations(caught)
    assert crosses is False


def test_raster_numpy_ufunc_no_deprecation_warning():
    grid = RasterGrid.from_bbox(
        xmin=-119.0, ymin=34.0, xmax=-118.0, ymax=35.0, cell_size=0.5, crs=WGS84
    )
    raster = Raster(np.linspace(0, np.pi, 4).reshape(2, 2), geometry=grid)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = np.cos(raster) + np.sin(raster)

    _assert_no_deprecations(caught)
    assert isinstance(result, Raster)
    assert result.shape == (2, 2)
