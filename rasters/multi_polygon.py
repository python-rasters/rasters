from __future__ import annotations

from typing import Union

import shapely

from .CRS import CRS, WGS84
from .bbox import BBox
from .vector_geometry import VectorGeometry, MultiVectorGeometry

class MultiPolygon(MultiVectorGeometry):
    def __init__(self, *args, crs: Union[CRS, str] = WGS84):
        if isinstance(args[0], MultiPolygon):
            geometry = args[0].geometry
            crs = args[0].crs
        else:
            geometry = shapely.geometry.MultiPolygon(*args)

        MultiVectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry

    @property
    def bbox(self) -> BBox:
        return BBox(*self.geometry.bounds, crs=self.crs)
