import numpy as np

def local_UTM_proj4_from_lat_lon(lat: float, lon: float) -> str:
    UTM_zone = (np.floor((lon + 180) / 6) % 60) + 1
    UTM_proj4 = f"+proj=utm +zone={UTM_zone} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    return UTM_proj4