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

def wrap_geometry(geometry: Any, crs: Union[CRS, str] = None) -> SpatialGeometry:
    if isinstance(geometry, SpatialGeometry):
        return geometry

    if crs is None:
        crs = WGS84

    if isinstance(geometry, str):
        geometry = shapely.geometry.shape(geometry)

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
