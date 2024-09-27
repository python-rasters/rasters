from __future__ import annotations

from abc import abstractmethod
from typing import Union, TYPE_CHECKING

import numpy as np

from .CRS import CRS, WGS84
from .local_UTM_proj4_from_lat_lon import local_UTM_proj4_from_lat_lon

if TYPE_CHECKING:
    from .bbox import BBox
    from .point import Point

class SpatialGeometry:
    def __init__(self, *args, crs: Union[CRS, str] = WGS84, **kwargs):
        if not isinstance(crs, CRS):
            crs = CRS(crs)

        self._crs = crs

    @property
    def crs(self) -> CRS:
        return self._crs

    @property
    @abstractmethod
    def bbox(self) -> BBox:
        pass

    @abstractmethod
    def to_crs(self, CRS: Union[CRS, "str"]) -> SpatialGeometry:
        pass

    @property
    def latlon(self) -> SpatialGeometry:
        return self.to_crs(WGS84)

    @property
    @abstractmethod
    def centroid(self) -> Point:
        raise NotImplementedError(f"centroid property not implemented by {self.__class__.__name__}")

    @property
    def centroid_latlon(self) -> Point:
        return self.centroid.latlon

    @property
    def local_UTM_proj4(self) -> str:
        centroid = self.centroid.latlon
        lat = centroid.y
        lon = centroid.x
        # UTM_zone = (np.floor((lon + 180) / 6) % 60) + 1
        # UTM_proj4 = f"+proj=utm +zone={UTM_zone} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs"

        UTM_proj4 = local_UTM_proj4_from_lat_lon(lat, lon)

        return UTM_proj4
