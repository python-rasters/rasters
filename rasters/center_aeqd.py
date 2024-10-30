from .CRS import CRS
from .point import Point

def center_aeqd(center_coord: Point) -> CRS:
    """
    Generate Azimuthal Equal Area CRS centered at given lat/lon.

    Args:
        center_coord (Point): Point object containing latitude and longitude of the center of the CRS.

    Returns:
        CRS: pyproj.CRS object of the centered CRS.
    """
    return CRS(f"+proj=aeqd +lat_0={center_coord.y} +lon_0={center_coord.x}")
