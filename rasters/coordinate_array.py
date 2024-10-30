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
    """
    A class representing an array of coordinates with a coordinate reference system (CRS).

    Attributes:
        x (np.ndarray): The x-coordinates.
        y (np.ndarray): The y-coordinates.
        crs (CRS): The coordinate reference system. Defaults to WGS84.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, crs: Union[CRS, str] = WGS84, **kwargs):
        """
        Initializes a new CoordinateArray object.

        Args:
            x (np.ndarray): The x-coordinates.
            y (np.ndarray): The y-coordinates.
            crs (Union[CRS, str]): The coordinate reference system. Defaults to WGS84.
            **kwargs: Additional keyword arguments passed to the parent class constructor.
        """
        super(CoordinateArray, self).__init__(crs=crs, **kwargs)
        self.x = x
        self.y = y

    def bbox(self) -> BBox:
        """
        Calculates the bounding box of the coordinate array.

        Returns:
            BBox: The bounding box of the coordinate array.
        """
        from .bbox import BBox  # Import here to avoid circular dependency
        return BBox.from_points(self.x, self.y, crs=self.crs)

    def centroid(self) -> Point:
        """
        Calculates the centroid of the coordinate array.

        Returns:
            Point: The centroid of the coordinate array.
        """
        return Point(np.nanmean(self.x), np.nanmean(self.y), crs=self.crs)
    
    def to_crs(self, crs: CRS | str) -> SpatialGeometry:
        """
        Transforms the coordinate array to a new CRS.

        Args:
            crs (CRS | str): The target CRS.

        Returns:
            CoordinateArray: A new CoordinateArray with the transformed coordinates.
        """
        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
        x, y = transformer.transform(self.x, self.y)
        result = CoordinateArray(x, y, crs=crs)

        return result
    
    @property
    def latlon(self) -> CoordinateArray:
        """
        Returns the coordinate array in the WGS84 (latitude/longitude) CRS.

        Returns:
            CoordinateArray: The coordinate array in WGS84.
        """
        return self.to_crs(WGS84)

    @property
    def lat(self) -> np.ndarray:
        """
        Returns the latitudes of the coordinate array in the WGS84 CRS.

        Returns:
            np.ndarray: The array of latitudes.
        """
        return self.latlon.y

    @property
    def lon(self) -> np.ndarray:
        """
        Returns the longitudes of the coordinate array in the WGS84 CRS.

        Returns:
            np.ndarray: The array of longitudes.
        """
        return self.latlon.x
