from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np
import shapely

from .CRS import CRS, WGS84
from .vector_geometry import VectorGeometry, SingleVectorGeometry

if TYPE_CHECKING:
    from .bbox import BBox
    from .point import Point

class Polygon(SingleVectorGeometry):
    def __init__(self, *args, crs: Union[CRS, str] = WGS84):
        if isinstance(args[0], Polygon):
            geometry = args[0].geometry
            crs = args[0].crs
        else:
            geometry = shapely.geometry.Polygon(*args)

        SingleVectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry

    @property
    def centroid(self) -> Point:
        from .point import Point
        """Returns the geometric center of the object"""
        return Point(self.geometry.centroid, crs=self.crs)

    @property
    def exterior(self):
        return self.geometry.exterior

    @property
    def is_empty(self):
        return self.geometry.is_empty

    @property
    def geom_type(self):
        return self.geometry.geom_type

    @property
    def bounds(self):
        return self.geometry.bounds

    @property
    def interiors(self):
        return self.geometry.interiors

    @property
    def wkt(self):
        return self.geometry.wkt

    @property
    def bbox(self) -> BBox:
        from .bbox import BBox
        x, y = self.exterior.xy
        x = np.array(x)
        y = np.array(y)
        x_min = float(np.nanmin(x))
        y_min = float(np.nanmin(y))
        x_max = float(np.nanmax(x))
        y_max = float(np.nanmax(y))
        bbox = BBox(x_min, y_min, x_max, y_max, crs=self.crs)

        return bbox
