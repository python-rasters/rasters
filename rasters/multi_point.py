from __future__ import annotations

from typing import Union

import numpy as np
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

    @property
    def x(self) -> np.ndarray:
        """
        Returns the x-coordinates of the points in the multi-point geometry.

        Returns:
            np.ndarray: An array of x-coordinates.
        """
        return np.array([point.x for point in self.geometry.geoms])
    
    @property
    def y(self) -> np.ndarray:
        """
        Returns the y-coordinates of the points in the multi-point geometry.

        Returns:
            np.ndarray: An array of y-coordinates.
        """
        return np.array([point.y for point in self.geometry.geoms])
    
    @property
    def xmin(self) -> float:
        """
        Returns the minimum x-coordinate of the points in the multi-point geometry.

        Returns:
            float: The minimum x-coordinate.
        """
        return np.nanmin(self.x)
    
    @property
    def ymin(self) -> float:
        """
        Returns the minimum y-coordinate of the points in the multi-point geometry.

        Returns:
            float: The minimum y-coordinate.
        """
        return np.nanmin(self.y)
    
    @property
    def xmax(self) -> float:
        """
        Returns the maximum x-coordinate of the points in the multi-point geometry.

        Returns:
            float: The maximum x-coordinate.
        """
        return np.nanmax(self.x)
    
    @property
    def ymax(self) -> float:
        """
        Returns the maximum y-coordinate of the points in the multi-point geometry.

        Returns:
            float: The maximum y-coordinate.
        """
        return np.nanmax(self.y)
    
    @property
    def bbox(self) -> "BBox":
        """
        Returns the bounding box of the multi-point geometry.

        Returns:
            BBox: The bounding box of the multi-point geometry.
        """
        from .bbox import BBox
        return BBox(self.xmin, self.ymin, self.xmax, self.ymax, crs=self.crs)