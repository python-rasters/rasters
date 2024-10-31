from __future__ import annotations

from typing import Union

import shapely

from .CRS import CRS, WGS84
from .vector_geometry import MultiVectorGeometry

class MultiPoint(MultiVectorGeometry):
    """
    A class representing a collection of points with a coordinate reference system (CRS).

    This class extends the MultiVectorGeometry class and uses shapely to represent 
    the multi-point geometry.

    Attributes:
        geometry (shapely.geometry.MultiPoint): The shapely geometry representing the multi-point.
        crs (CRS): The coordinate reference system of the multi-point. Defaults to WGS84.
    """
    def __init__(self, points, crs: Union[CRS, str] = WGS84):
        """
        Initializes a new MultiPoint object.

        Args:
            points: A list of point coordinates or a MultiPoint object.
                    If a MultiPoint object is provided, its geometry and CRS are used.
            crs (Union[CRS, str], optional): The coordinate reference system. Defaults to WGS84.
        """
        if isinstance(points[0], MultiPoint):
            # If the input is a MultiPoint, use its geometry and CRS
            geometry = points[0].geometry
            crs = points[0].crs
        else:
            # Otherwise, create a new shapely MultiPoint from the coordinates
            geometry = shapely.geometry.MultiPoint(points)

        # Initialize the parent class with the CRS
        MultiVectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry
