from __future__ import annotations

from typing import Union

import shapely

from .CRS import CRS, WGS84
from .bbox import BBox
from .vector_geometry import MultiVectorGeometry


class MultiPolygon(MultiVectorGeometry):
    """
    A class representing a collection of polygons with a coordinate reference system (CRS).

    This class extends the MultiVectorGeometry class and uses shapely to represent
    the multi-polygon geometry.

    Attributes:
        geometry (shapely.geometry.MultiPolygon): The shapely geometry representing the multi-polygon.
        crs (CRS): The coordinate reference system of the multi-polygon. Defaults to WGS84.
    """
    def __init__(self, *args, crs: Union[CRS, str] = WGS84):
        """
        Initializes a new MultiPolygon object.

        Args:
            *args:  A MultiPolygon object or arguments to pass to shapely.geometry.MultiPolygon.
                    If a MultiPolygon object is provided, its geometry and CRS are used.
            crs (Union[CRS, str], optional): The coordinate reference system. Defaults to WGS84.
        """
        if isinstance(args[0], MultiPolygon):
            # If the input is a MultiPolygon, use its geometry and CRS
            geometry = args[0].geometry
            crs = args[0].crs
        else:
            # Otherwise, create a new shapely MultiPolygon from the arguments
            geometry = shapely.geometry.MultiPolygon(*args)

        # Initialize the parent class with the CRS
        MultiVectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry

    @property
    def bbox(self) -> BBox:
        """
        Calculates the bounding box of the multi-polygon.

        Returns:
            BBox: The bounding box of the multi-polygon.
        """
        return BBox(*self.geometry.bounds, crs=self.crs)
