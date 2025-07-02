from __future__ import annotations

from typing import Union, Optional, TYPE_CHECKING
import numpy as np
import shapely
from pyproj import Transformer

from .CRS import CRS, WGS84
from .vector_geometry import MultiVectorGeometry
from .point import Point # Assuming you have a Point class

if TYPE_CHECKING:
    from .bbox import BBox

class MultiPoint(MultiVectorGeometry):
    """
    A class representing a collection of points with a coordinate reference system (CRS).

    This class extends the MultiVectorGeometry class and uses shapely to represent
    the multi-point geometry. It also incorporates functionality to manage
    coordinate arrays and perform CRS transformations.

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
                # Ensure input points are in a format shapely can handle (e.g., list of tuples/lists)
                if isinstance(points, np.ndarray):
                    # If it's a NumPy array, it's likely (N, 2) or (N, 3)
                    geometry = shapely.geometry.MultiPoint(points)
                else:
                    # Assume it's an iterable of point-like structures
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
        # Ensure coordinates are extracted correctly, especially for empty geometries
        if self.geometry.is_empty:
            return np.array([])
        return np.array([point.x for point in self.geometry.geoms])
    
    @property
    def y(self) -> np.ndarray:
        """
        Returns the y-coordinates of the points in the multi-point geometry.

        Returns:
            np.ndarray: An array of y-coordinates.
        """
        if self.geometry.is_empty:
            return np.array([])
        return np.array([point.y for point in self.geometry.geoms])
    
    @property
    def z(self) -> np.ndarray:
        """
        Returns the z-coordinates of the points in the multi-point geometry.

        Returns:
            np.ndarray: An array of z-coordinates. Returns empty array if no z-coords.
        """
        if self.geometry.is_empty:
            return np.array([])
        # Shapely's .coords returns a sequence of (x, y, z) or (x, y) tuples
        # We need to explicitly check if z exists for each point or rely on overall dimension
        coords = np.array(self.geometry.coords)
        if coords.shape[1] == 3:
            return coords[:, 2]
        return np.array([]) # Return empty if no Z dimension

    @property
    def xmin(self) -> float:
        """
        Returns the minimum x-coordinate of the points in the multi-point geometry.

        Returns:
            float: The minimum x-coordinate.
        """
        # Use geometry.bounds for efficiency if possible, or handle empty explicitly
        if self.geometry.is_empty:
            return np.nan
        return self.geometry.bounds[0] # xmin
    
    @property
    def ymin(self) -> float:
        """
        Returns the minimum y-coordinate of the points in the multi-point geometry.

        Returns:
            float: The minimum y-coordinate.
        """
        if self.geometry.is_empty:
            return np.nan
        return self.geometry.bounds[1] # ymin
    
    @property
    def xmax(self) -> float:
        """
        Returns the maximum x-coordinate of the points in the multi-point geometry.

        Returns:
            float: The maximum x-coordinate.
        """
        if self.geometry.is_empty:
            return np.nan
        return self.geometry.bounds[2] # xmax
    
    @property
    def ymax(self) -> float:
        """
        Returns the maximum y-coordinate of the points in the multi-point geometry.

        Returns:
            float: The maximum y-coordinate.
        """
        if self.geometry.is_empty:
            return np.nan
        return self.geometry.bounds[3] # ymax
    
    @property
    def bbox(self) -> BBox:
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

    def centroid(self) -> Point:
        """
        Calculates the centroid of the multi-point geometry.
        For a MultiPoint, this is typically the mean of all constituent points' coordinates.

        Returns:
            Point: The centroid of the multi-point.
        """
        if self.geometry.is_empty:
            return Point(np.nan, np.nan, crs=self.crs) # Centroid of empty set is undefined

        # Shapely's .centroid for MultiPoint would be the centroid of its convex hull sometimes
        # or the average of all points. Here, we emulate CoordinateArray's mean
        # to ensure consistency with what it offered.
        # This is equivalent to shapely.geometry.MultiPoint.centroid.x/y for a simple average
        # for Points.
        return Point(self.geometry.centroid.x, self.geometry.centroid.y, crs=self.crs)
    
    def to_crs(self, crs: Union[CRS, str]) -> "MultiPoint":
        """
        Transforms the multi-point to a new CRS.

        Args:
            crs (CRS | str): The target CRS.

        Returns:
            MultiPoint: A new MultiPoint with the transformed coordinates.
        """
        if isinstance(crs, str):
            crs = CRS(crs)

        if self.crs.equals(crs):
            return self # No transformation needed

        # Handle empty geometry gracefully
        if self.geometry.is_empty:
            return MultiPoint(crs=crs)

        # Extract coordinates from shapely geometry
        # .coords returns a sequence of (x, y) or (x, y, z) tuples
        coords_array = np.array(self.geometry.coords)

        # Assuming your CRS class has a method to_pyproj() that returns a pyproj.CRS object
        transformer = Transformer.from_crs(self.crs.to_pyproj(), crs.to_pyproj(), always_xy=True)

        if coords_array.shape[1] == 3: # If Z coordinates exist
            x_transformed, y_transformed, z_transformed = transformer.transform(
                coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]
            )
            transformed_coords = np.column_stack((x_transformed, y_transformed, z_transformed))
        else: # 2D coordinates
            x_transformed, y_transformed = transformer.transform(
                coords_array[:, 0], coords_array[:, 1]
            )
            transformed_coords = np.column_stack((x_transformed, y_transformed))

        # Create a new MultiPoint from transformed coordinates
        return MultiPoint(transformed_coords, crs=crs)
    
    @property
    def latlon(self) -> "MultiPoint": # Changed return type to MultiPoint
        """
        Returns the multi-point in the WGS84 (latitude/longitude) CRS.

        Returns:
            MultiPoint: The multi-point in WGS84.
        """
        return self.to_crs(WGS84)

    @property
    def lat(self) -> np.ndarray:
        """
        Returns the latitudes of the points in the WGS84 CRS.

        Returns:
            np.ndarray: The array of latitudes.
        """
        # Perform on-the-fly transformation to WGS84 and extract Y (latitude)
        return self.latlon.y

    @property
    def lon(self) -> np.ndarray:
        """
        Returns the longitudes of the points in the WGS84 CRS.

        Returns:
            np.ndarray: The array of longitudes.
        """
        # Perform on-the-fly transformation to WGS84 and extract X (longitude)
        return self.latlon.x
