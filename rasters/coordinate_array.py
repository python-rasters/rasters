from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np
from pyproj import Transformer

from .CRS import CRS, WGS84
from .bbox import BBox
from .point import Point
from .spatial_geometry import SpatialGeometry

if TYPE_CHECKING:
    from .bbox import BBox

class CoordinateArray(SpatialGeometry):
    def __init__(self, x: np.ndarray, y: np.ndarray, crs: Union[CRS, str] = WGS84, **kwargs):
        super(CoordinateArray, self).__init__(crs=crs, **kwargs)
        self.x = x
        self.y = y

    def bbox(self) -> BBox:
        from .bbox import BBox
        return BBox.from_points(self.x, self.y, crs=self.crs)

    def centroid(self) -> Point:
        return Point(np.nanmean(self.x), np.nanmean(self.y), crs=self.crs)
    
    def to_crs(self, crs: CRS | str) -> SpatialGeometry:
        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
        x, y = transformer.transform(self.x, self.y)
        result = CoordinateArray(x, y, crs=crs)

        return result
    
    @property
    def latlon(self) -> CoordinateArray:
        return self.to_crs(WGS84)

    @property
    def lat(self) -> np.ndarray:
        """
        array of latitudes
        """
        return self.latlon.y

    @property
    def lon(self) -> np.ndarray:
        """
        array of longitudes
        """
        return self.latlon.x
