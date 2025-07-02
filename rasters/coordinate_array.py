from __future__ import annotations

from typing import Union, TYPE_CHECKING
import numpy as np
import warnings # Import the warnings module

from .CRS import CRS, WGS84
from .spatial_geometry import SpatialGeometry

if TYPE_CHECKING:
    from .bbox import BBox

class CoordinateArray(SpatialGeometry):
    """
    DEPRECATED: This class is deprecated and will be removed in a future version.
    Please use the `MultiPoint` class instead, which now includes all the
    functionality previously provided by `CoordinateArray`, along with
    Shapely-based geometric operations.

    A class representing an array of coordinates with a coordinate reference system (CRS).

    Attributes:
        x (np.ndarray): The x-coordinates.
        y (np.ndarray): The y-coordinates.
        crs (CRS): The coordinate reference system. Defaults to WGS84.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, crs: Union[CRS, str] = WGS84, **kwargs):
        warnings.warn(
            "The `CoordinateArray` class is deprecated and will be removed in a future version. "
            "Please use the `MultiPoint` class instead, which now includes all `CoordinateArray` functionality.",
            DeprecationWarning,
            stacklevel=2 # Points to the user's code, not this internal method
        )
        super(CoordinateArray, self).__init__(crs=crs, **kwargs)
        
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if x.shape != y.shape:
            raise ValueError("x and y arrays must have the same shape.")
        if x.ndim != 1:
            raise ValueError("x and y arrays must be 1-dimensional.")

        self.x = x
        self.y = y

    def bbox(self) -> BBox:
        """
        Calculates the bounding box of the coordinate array.

        Returns:
            BBox: The bounding box of the coordinate array.
        """
        from .bbox import BBox  # Import here to avoid circular dependency
        if self.x.size == 0:
            return BBox(crs=self.crs) # Return empty BBox for empty CoordinateArray
        return BBox.from_points(self.x, self.y, crs=self.crs)

    def centroid(self) -> "Point": # Assuming Point is defined elsewhere and is callable
        """
        Calculates the centroid of the coordinate array.

        Returns:
            Point: The centroid of the coordinate array.
        """
        # Ensure Point is imported or type-hinted if it's in a separate module
        from .point import Point 
        if self.x.size == 0:
            return Point(np.nan, np.nan, crs=self.crs)
        return Point(np.nanmean(self.x), np.nanmean(self.y), crs=self.crs)
    
    def to_crs(self, crs: Union[CRS, str]) -> "CoordinateArray":
        """
        Transforms the coordinate array to a new CRS.

        Args:
            crs (CRS | str): The target CRS.

        Returns:
            CoordinateArray: A new CoordinateArray with the transformed coordinates.
        """
        from pyproj import Transformer # Import here to avoid circular dependency if not already at top
        
        if isinstance(crs, str):
            crs = CRS(crs)

        if self.crs.equals(crs):
            return self

        if self.x.size == 0:
            return CoordinateArray(np.array([]), np.array([]), crs=crs)

        # Assuming your CRS class has a method to_pyproj() that returns a pyproj.CRS object
        transformer = Transformer.from_crs(self.crs.to_pyproj(), crs.to_pyproj(), always_xy=True)
        x, y = transformer.transform(self.x, self.y)
        result = CoordinateArray(x, y, crs=crs)

        return result
    
    @property
    def latlon(self) -> "CoordinateArray":
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
