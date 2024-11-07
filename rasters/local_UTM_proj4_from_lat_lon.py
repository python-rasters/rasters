import numpy as np

def local_UTM_proj4_from_lat_lon(lat: float, lon: float) -> str:
    """
    Generate a local UTM projection string (proj4 format) based on given latitude and longitude.

    This function calculates the UTM zone based on the provided longitude and 
    constructs a proj4 string for the corresponding UTM projection.

    Args:
        lat (float): The latitude of the point.
        lon (float): The longitude of the point.

    Returns:
        str: The proj4 string for the local UTM projection.
    """
    UTM_zone = (np.floor((lon + 180) / 6) % 60) + 1
    UTM_proj4 = f"+proj=utm +zone={int(UTM_zone)} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    return UTM_proj4
