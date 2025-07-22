from __future__ import annotations

from typing import Any, Union, TYPE_CHECKING

import shapely

from .CRS import CRS
from .spatial_geometry import SpatialGeometry

if TYPE_CHECKING:
    from .point import Point
    from .multi_point import MultiPoint
    from .polygon import Polygon
    from .multi_polygon import MultiPolygon

WGS84 = CRS.from_epsg(4326)  # Define WGS84 CRS

def wrap_geometry(geometry: Any, crs: Union[CRS, str] = None) -> SpatialGeometry:
    """
    Converts a given geometry into a SpatialGeometry object.

    This function handles various geometry types, including:
     - Existing SpatialGeometry objects: Returned as is.
     - Shapely geometry objects: Converted to the corresponding SpatialGeometry type.
     - GeoJSON strings: Parsed and converted to SpatialGeometry.

    Args:
        geometry: The geometry to convert. Can be a SpatialGeometry object,
                  a shapely geometry object, or a GeoJSON string.
        crs: The coordinate reference system of the geometry. If None, WGS84 is used.
             Can be a CRS object or a string representation of a CRS.

    Returns:
        A SpatialGeometry object representing the input geometry.

    Raises:
        ValueError: If the geometry type is not supported.
    """
    from .point import Point
    from .multi_point import MultiPoint
    from .polygon import Polygon
    from .multi_polygon import MultiPolygon
    from .raster_geometry import RasterGeometry

    if isinstance(geometry, RasterGeometry):
        # If the geometry is a RasterGeometry, return it as is
        return geometry

    # Check if the geometry is already a SpatialGeometry by checking for specific types
    if isinstance(geometry, (Point, MultiPoint, Polygon, MultiPolygon)):
        # If the geometry is already a SpatialGeometry, return it as is
        return geometry

    if crs is None:
        # Default to WGS84 if no CRS is provided
        crs = WGS84
    elif isinstance(crs, str):
        # Convert string representation of CRS to CRS object
        crs = CRS.from_string(crs)

    if isinstance(geometry, str):
        # Parse GeoJSON string into a shapely geometry object
        geometry = shapely.geometry.shape(geometry)

    # Convert shapely geometry objects to corresponding SpatialGeometry types
    if isinstance(geometry, shapely.geometry.Point):
        return Point(geometry, crs=crs)
    elif isinstance(geometry, shapely.geometry.MultiPoint):
        return MultiPoint(geometry, crs=crs)
    elif isinstance(geometry, shapely.geometry.Polygon):
        return Polygon(geometry, crs=crs)
    elif isinstance(geometry, shapely.geometry.MultiPolygon):
        return MultiPolygon(geometry, crs=crs)
    else:
        raise ValueError(f"unsupported geometry type: {type(geometry)}")