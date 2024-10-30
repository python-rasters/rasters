from typing import Union

import math
import shapely

from .CRS import CRS
from .point import Point

def local_UTM_proj4(cls, point_latlon: Union[Point, str]) -> CRS:
    """
    Generate a local UTM projection based on a given point.

    Args:
        point_latlon (Union[Point, str]): Point object or WKT string containing latitude and longitude.

    Returns:
        CRS: pyproj.CRS object of the local UTM projection.
    """
    try:
        if isinstance(point_latlon, str):
            point_latlon = shapely.wkt.loads(point_latlon)
    except Exception as e:
        raise ValueError(f"Invalid WKT string: {e}")

    lat = point_latlon.y
    lon = point_latlon.x

    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")

    UTM_zone = (math.floor((lon + 180) / 6) % 60) + 1
    UTM_proj4 = CRS(
        f"+proj=utm +zone={UTM_zone} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )

    return UTM_proj4
