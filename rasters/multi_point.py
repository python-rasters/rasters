from __future__ import annotations

from typing import Union

import shapely

from .CRS import CRS, WGS84
from .vector_geometry import MultiVectorGeometry

class MultiPoint(MultiVectorGeometry):
    def __init__(self, points, crs: Union[CRS, str] = WGS84):
        if isinstance(points[0], MultiPoint):
            geometry = points[0].geometry
            crs = points[0].crs
        else:
            geometry = shapely.geometry.MultiPoint(points)

        MultiVectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry
