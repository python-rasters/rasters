from __future__ import annotations

from typing import Union, Optional
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
    def __init__(
        self,
        points: Optional[Union[list, tuple, np.ndarray, MultiPoint]] = None,
        x: Optional[Union[list, np.ndarray]] = None,
        y: Optional[Union[list, np.ndarray]] = None,
        crs: Union[CRS, str] = WGS84
    ):
        """
        Initializes a new MultiPoint object.

        Args:
            points: An optional list of point coordinates, a numpy array of coordinates,
                    or a MultiPoint object. If provided, 'x' and 'y' arguments are ignored.
            x (Optional[Union[list, np.ndarray]]): An optional array or list of x-coordinates.
                                                    Must be provided with 'y' if 'points' is not used.
            y (Optional[Union[list, np.ndarray]]): An optional array or list of y-coordinates.
                                                    Must be provided with 'x' if 'points' is not used.
            crs (Union[CRS, str], optional): The coordinate reference system. Defaults to WGS84.

        Raises:
            ValueError: If an invalid combination of arguments is provided (e.g., only x or only y,
                        or both points and x/y are provided).
        """
        if points is not None and (x is not None or y is not None):
            raise ValueError("Cannot provide both 'points' and 'x'/'y' arguments.")
        
        if x is not None and y is None:
            raise ValueError("If 'x' is provided, 'y' must also be provided.")

        if y is not None and x is None:
            raise ValueError("If 'y' is provided, 'x' must also be provided.")

        geometry = None
        if points is not None:
            if isinstance(points, MultiPoint):
                # If the input is a MultiPoint, use its geometry and CRS
                geometry = points.geometry
                crs = points.crs
            else:
                # Otherwise, create a new shapely MultiPoint from the coordinates
                geometry = shapely.geometry.MultiPoint(points)
        elif x is not None and y is not None:
            if len(x) != len(y):
                raise ValueError("Length of 'x' array must match length of 'y' array.")
            coords = np.column_stack((x, y))
            geometry = shapely.geometry.MultiPoint(coords)
        else:
            # Initialize with an empty MultiPoint if no points are provided
            geometry = shapely.geometry.MultiPoint()

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
        # Handle empty geometry case
        if self.geometry.is_empty:
            return np.nan
        return np.nanmin(self.x)
    
    @property
    def ymin(self) -> float:
        """
        Returns the minimum y-coordinate of the points in the multi-point geometry.

        Returns:
            float: The minimum y-coordinate.
        """
        # Handle empty geometry case
        if self.geometry.is_empty:
            return np.nan
        return np.nanmin(self.y)
    
    @property
    def xmax(self) -> float:
        """
        Returns the maximum x-coordinate of the points in the multi-point geometry.

        Returns:
            float: The maximum x-coordinate.
        """
        # Handle empty geometry case
        if self.geometry.is_empty:
            return np.nan
        return np.nanmax(self.x)
    
    @property
    def ymax(self) -> float:
        """
        Returns the maximum y-coordinate of the points in the multi-point geometry.

        Returns:
            float: The maximum y-coordinate.
        """
        # Handle empty geometry case
        if self.geometry.is_empty:
            return np.nan
        return np.nanmax(self.y)
    
    @property
    def bbox(self) -> "BBox":
        """
        Returns the bounding box of the multi-point geometry.

        Returns:
            BBox: The bounding box of the multi-point geometry.
        """
        from .bbox import BBox
        # If the geometry is empty, create an empty BBox
        if self.geometry.is_empty:
            return BBox(crs=self.crs)
        return BBox(self.xmin, self.ymin, self.xmax, self.ymax, crs=self.crs)