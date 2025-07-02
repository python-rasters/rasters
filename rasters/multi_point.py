from __future__ import annotations

from typing import Union, Optional, TYPE_CHECKING
import numpy as np
import shapely
from pyproj import Transformer

from .CRS import CRS, WGS84
from .vector_geometry import MultiVectorGeometry

# TYPE_CHECKING imports to avoid circular imports at runtime
if TYPE_CHECKING:
    from .bbox import BBox
    from .point import Point

class MultiPoint(MultiVectorGeometry):
    """
    A class representing a collection of points with coordinate reference system (CRS) support.

    This class provides a wrapper around shapely.geometry.MultiPoint with additional
    functionality for coordinate reference system handling, transformations, and
    geometric operations specific to geospatial applications involving multiple points.

    The MultiPoint class is particularly useful for:
    - Managing collections of GPS coordinates or survey points
    - Performing batch coordinate transformations
    - Spatial analysis involving multiple point locations
    - Converting between different coordinate reference systems

    Inherits from MultiVectorGeometry, which provides base functionality for
    multi-geometry objects with CRS support.

    Attributes:
        geometry (shapely.geometry.MultiPoint): The underlying shapely geometry representing the collection of points.
        crs (CRS): The coordinate reference system of the multi-point geometry.

    Example:
        >>> # Create from coordinate arrays
        >>> x_coords = [10, 20, 30]
        >>> y_coords = [15, 25, 35]
        >>> mp = MultiPoint(x=x_coords, y=y_coords, crs=WGS84)
        
        >>> # Create from point list
        >>> points = [(10, 15), (20, 25), (30, 35)]
        >>> mp = MultiPoint(points=points, crs=WGS84)
        
        >>> # Access coordinates
        >>> print(mp.x)  # [10, 20, 30]
        >>> print(mp.y)  # [15, 25, 35]
    """
    def __init__(
        self,
        points: Optional[Union[list, tuple, np.ndarray, MultiPoint]] = None,
        x: Optional[Union[list, np.ndarray]] = None,
        y: Optional[Union[list, np.ndarray]] = None,
        crs: Union[CRS, str] = WGS84
    ):
        """
        Initialize a MultiPoint geometry with coordinate reference system support.

        Supports multiple construction patterns:
        1. From a list/array of point coordinates
        2. From separate x and y coordinate arrays
        3. From another MultiPoint object (copy constructor)
        4. Empty initialization for later population

        Args:
            points (Optional[Union[list, tuple, np.ndarray, MultiPoint]]): 
                An optional collection of point coordinates. Can be:
                - List of tuples: [(x1, y1), (x2, y2), ...]
                - NumPy array: shape (N, 2) or (N, 3) for 2D/3D points
                - Another MultiPoint object (copies geometry and CRS)
                If provided, 'x' and 'y' arguments are ignored.
                
            x (Optional[Union[list, np.ndarray]]): Array or list of x-coordinates.
                Must be provided with 'y' if 'points' is not used.
                
            y (Optional[Union[list, np.ndarray]]): Array or list of y-coordinates.
                Must be provided with 'x' if 'points' is not used.
                
            crs (Union[CRS, str], optional): The coordinate reference system. 
                Defaults to WGS84. Ignored if copying from another MultiPoint.

        Raises:
            ValueError: If an invalid combination of arguments is provided:
                - Both 'points' and 'x'/'y' are provided
                - Only 'x' or only 'y' is provided (both required)
                - Length mismatch between 'x' and 'y' arrays

        Example:
            >>> # From coordinate arrays
            >>> mp1 = MultiPoint(x=[1, 2, 3], y=[4, 5, 6])
            
            >>> # From point list
            >>> mp2 = MultiPoint(points=[(1, 4), (2, 5), (3, 6)])
            
            >>> # Copy constructor
            >>> mp3 = MultiPoint(points=mp1)
            
            >>> # Empty initialization
            >>> mp4 = MultiPoint()
        """
        # Validate argument combinations to prevent ambiguous initialization
        if points is not None and (x is not None or y is not None):
            raise ValueError("Cannot provide both 'points' and 'x'/'y' arguments.")
        
        if x is not None and y is None:
            raise ValueError("If 'x' is provided, 'y' must also be provided.")

        if y is not None and x is None:
            raise ValueError("If 'y' is provided, 'x' must also be provided.")

        geometry = None
        
        if points is not None:
            if isinstance(points, MultiPoint):
                # Copy constructor: extract geometry and CRS from existing MultiPoint
                geometry = points.geometry
                crs = points.crs
            else:
                # Create new shapely MultiPoint from coordinate data
                if isinstance(points, np.ndarray):
                    # Handle NumPy arrays with shape (N, 2) or (N, 3) for 2D/3D points
                    geometry = shapely.geometry.MultiPoint(points)
                else:
                    # Handle lists, tuples, or other iterables of point-like structures
                    geometry = shapely.geometry.MultiPoint(points)

        elif x is not None and y is not None:
            # Create from separate coordinate arrays
            if len(x) != len(y):
                raise ValueError("Length of 'x' array must match length of 'y' array.")
            
            # Stack coordinates into (N, 2) array format expected by shapely
            coords = np.column_stack((x, y))
            geometry = shapely.geometry.MultiPoint(coords)
        else:
            # Initialize with an empty MultiPoint for later population
            geometry = shapely.geometry.MultiPoint()

        # Initialize the parent MultiVectorGeometry class with the specified CRS
        MultiVectorGeometry.__init__(self, crs=crs)

        # Store the shapely geometry object
        self.geometry = geometry

    @property
    def x(self) -> np.ndarray:
        """
        Get the x-coordinates (longitude or easting) of all points in the collection.
        
        The interpretation of x-coordinates depends on the coordinate reference system:
        - In geographic CRS (like WGS84): longitude values in decimal degrees
        - In projected CRS (like UTM): easting values in meters

        Returns:
            np.ndarray: Array of x-coordinate values. Returns empty array if geometry is empty.
            
        Example:
            >>> mp = MultiPoint(x=[10, 20, 30], y=[15, 25, 35])
            >>> print(mp.x)  # [10. 20. 30.]
        """
        # Handle empty geometries gracefully to avoid errors
        if self.geometry.is_empty:
            return np.array([])
        
        # Extract x-coordinates from all constituent Point geometries
        return np.array([point.x for point in self.geometry.geoms])
    
    @property
    def y(self) -> np.ndarray:
        """
        Get the y-coordinates (latitude or northing) of all points in the collection.
        
        The interpretation of y-coordinates depends on the coordinate reference system:
        - In geographic CRS (like WGS84): latitude values in decimal degrees
        - In projected CRS (like UTM): northing values in meters

        Returns:
            np.ndarray: Array of y-coordinate values. Returns empty array if geometry is empty.
            
        Example:
            >>> mp = MultiPoint(x=[10, 20, 30], y=[15, 25, 35])
            >>> print(mp.y)  # [15. 25. 35.]
        """
        # Handle empty geometries gracefully to avoid errors
        if self.geometry.is_empty:
            return np.array([])
        
        # Extract y-coordinates from all constituent Point geometries
        return np.array([point.y for point in self.geometry.geoms])
    
    @property
    def z(self) -> np.ndarray:
        """
        Get the z-coordinates (elevation or height) of all points in the collection.
        
        Z-coordinates represent the third dimension, typically elevation above
        sea level, height, or depth depending on the coordinate reference system.
        Not all coordinate systems or point collections include z-coordinates.

        Returns:
            np.ndarray: Array of z-coordinate values. Returns empty array if:
                - The geometry is empty
                - The points are 2D only (no z-dimension)
                
        Example:
            >>> # 3D points with elevation data
            >>> points_3d = [(10, 15, 100), (20, 25, 150), (30, 35, 200)]
            >>> mp = MultiPoint(points=points_3d)
            >>> print(mp.z)  # [100. 150. 200.]
        """
        # Handle empty geometries gracefully
        if self.geometry.is_empty:
            return np.array([])
        
        # Extract coordinate array from shapely geometry
        # .coords returns a sequence of (x, y) or (x, y, z) tuples depending on dimensionality
        coords = np.array(self.geometry.coords)
        
        # Check if z-dimension exists (3D coordinates)
        if coords.shape[1] == 3:
            return coords[:, 2]  # Return z-coordinates (third column)
        
        # Return empty array if no z-dimension exists
        return np.array([])

    @property
    def xmin(self) -> float:
        """
        Get the minimum x-coordinate (westernmost longitude or smallest easting) of all points.
        
        This represents the western boundary of the point collection in geographic
        coordinates, or the smallest easting value in projected coordinates.

        Returns:
            float: The minimum x-coordinate value. Returns NaN if geometry is empty.
            
        Example:
            >>> mp = MultiPoint(x=[10, 20, 5], y=[15, 25, 35])
            >>> print(mp.xmin)  # 5.0
        """
        # Handle empty geometries by returning NaN to indicate undefined bounds
        if self.geometry.is_empty:
            return np.nan
        
        # Use shapely's efficient bounds property: (xmin, ymin, xmax, ymax)
        return self.geometry.bounds[0]
    
    @property
    def ymin(self) -> float:
        """
        Get the minimum y-coordinate (southernmost latitude or smallest northing) of all points.
        
        This represents the southern boundary of the point collection in geographic
        coordinates, or the smallest northing value in projected coordinates.

        Returns:
            float: The minimum y-coordinate value. Returns NaN if geometry is empty.
            
        Example:
            >>> mp = MultiPoint(x=[10, 20, 5], y=[15, 25, 5])
            >>> print(mp.ymin)  # 5.0
        """
        # Handle empty geometries by returning NaN to indicate undefined bounds
        if self.geometry.is_empty:
            return np.nan
        
        # Use shapely's efficient bounds property: (xmin, ymin, xmax, ymax)
        return self.geometry.bounds[1]
    
    @property
    def xmax(self) -> float:
        """
        Get the maximum x-coordinate (easternmost longitude or largest easting) of all points.
        
        This represents the eastern boundary of the point collection in geographic
        coordinates, or the largest easting value in projected coordinates.

        Returns:
            float: The maximum x-coordinate value. Returns NaN if geometry is empty.
            
        Example:
            >>> mp = MultiPoint(x=[10, 20, 5], y=[15, 25, 35])
            >>> print(mp.xmax)  # 20.0
        """
        # Handle empty geometries by returning NaN to indicate undefined bounds
        if self.geometry.is_empty:
            return np.nan
        
        # Use shapely's efficient bounds property: (xmin, ymin, xmax, ymax)
        return self.geometry.bounds[2]
    
    @property
    def ymax(self) -> float:
        """
        Get the maximum y-coordinate (northernmost latitude or largest northing) of all points.
        
        This represents the northern boundary of the point collection in geographic
        coordinates, or the largest northing value in projected coordinates.

        Returns:
            float: The maximum y-coordinate value. Returns NaN if geometry is empty.
            
        Example:
            >>> mp = MultiPoint(x=[10, 20, 5], y=[15, 25, 35])
            >>> print(mp.ymax)  # 35.0
        """
        # Handle empty geometries by returning NaN to indicate undefined bounds
        if self.geometry.is_empty:
            return np.nan
        
        # Use shapely's efficient bounds property: (xmin, ymin, xmax, ymax)
        return self.geometry.bounds[3]
    
    @property
    def bbox(self) -> "BBox":
        """
        Get the bounding box (envelope) that contains all points in the collection.
        
        The bounding box is the smallest rectangle that completely contains all
        points in the MultiPoint geometry. It's defined by the minimum and maximum
        x and y coordinates of all constituent points.

        Returns:
            BBox: A bounding box object with the same CRS as the MultiPoint.
                  Returns an empty BBox if the geometry is empty.
                  
        Example:
            >>> mp = MultiPoint(x=[10, 20, 5], y=[15, 25, 5])
            >>> bbox = mp.bbox
            >>> # bbox represents the rectangle (5, 5, 20, 25)
        """
        # Import here to avoid circular imports
        from .bbox import BBox
        
        # Handle empty geometry by creating an empty BBox with the same CRS
        if self.geometry.is_empty:
            return BBox(crs=self.crs)
        
        # Create BBox from the bounds of all points
        return BBox(self.xmin, self.ymin, self.xmax, self.ymax, crs=self.crs)

    def centroid(self) -> "Point":
        """
        Calculate the centroid (geometric center) of all points in the collection.
        
        For a MultiPoint geometry, the centroid is computed as the arithmetic mean
        of all constituent points' coordinates. This provides the "center of mass"
        if all points had equal weight.

        Returns:
            Point: A Point object representing the centroid with the same CRS as the MultiPoint.
                   Returns a Point with NaN coordinates if the geometry is empty.
                   
        Example:
            >>> mp = MultiPoint(x=[0, 10, 20], y=[0, 10, 20])
            >>> centroid = mp.centroid()
            >>> print(f"Centroid: ({centroid.x}, {centroid.y})")  # (10.0, 10.0)
        """
        # Import here to avoid circular imports
        from .point import Point
        
        # Handle empty geometry by returning a Point with undefined coordinates
        if self.geometry.is_empty:
            return Point(np.nan, np.nan, crs=self.crs)

        # Use shapely's built-in centroid calculation which computes the arithmetic mean
        # of all constituent point coordinates for MultiPoint geometries
        centroid_geom = self.geometry.centroid
        
        # Create and return a new Point with the centroid coordinates and same CRS
        return Point(centroid_geom.x, centroid_geom.y, crs=self.crs)
    
    def to_crs(self, crs: Union[CRS, str]) -> "MultiPoint":
        """
        Transform the MultiPoint to a different coordinate reference system.
        
        This method performs coordinate transformation using PyProj, which provides
        accurate transformations between different coordinate reference systems.
        The transformation preserves the geometric relationships between points
        while changing their coordinate values to match the target CRS.

        Args:
            crs (Union[CRS, str]): The target coordinate reference system.
                                   Can be a CRS object or string representation (e.g., "EPSG:4326").

        Returns:
            MultiPoint: A new MultiPoint object with coordinates transformed to the target CRS.
                        The original MultiPoint remains unchanged.
                        
        Example:
            >>> # Transform from WGS84 to UTM Zone 33N
            >>> wgs84_points = MultiPoint(x=[-1, 0, 1], y=[50, 51, 52], crs="EPSG:4326")
            >>> utm_points = wgs84_points.to_crs("EPSG:32633")
            >>> print(utm_points.crs)  # UTM Zone 33N
        """
        # Convert string CRS to CRS object if necessary
        if isinstance(crs, str):
            crs = CRS(crs)

        # No transformation needed if source and target CRS are the same
        if self.crs.equals(crs):
            return self

        # Handle empty geometry gracefully - return empty MultiPoint with target CRS
        if self.geometry.is_empty:
            return MultiPoint(crs=crs)

        # Extract coordinate array from shapely geometry
        # .coords returns a sequence of (x, y) or (x, y, z) tuples
        coords_array = np.array(self.geometry.coords)

        # Create PyProj transformer for coordinate transformation
        # always_xy=True ensures consistent x,y coordinate order regardless of CRS axis order
        transformer = Transformer.from_crs(
            self.crs.to_pyproj(), 
            crs.to_pyproj(), 
            always_xy=True
        )

        # Perform transformation based on coordinate dimensionality
        if coords_array.shape[1] == 3:
            # Handle 3D coordinates (x, y, z)
            x_transformed, y_transformed, z_transformed = transformer.transform(
                coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]
            )
            transformed_coords = np.column_stack((x_transformed, y_transformed, z_transformed))
        else:
            # Handle 2D coordinates (x, y)
            x_transformed, y_transformed = transformer.transform(
                coords_array[:, 0], coords_array[:, 1]
            )
            transformed_coords = np.column_stack((x_transformed, y_transformed))

        # Create and return a new MultiPoint with transformed coordinates
        return MultiPoint(transformed_coords, crs=crs)
    
    @property
    def latlon(self) -> "MultiPoint":
        """
        Transform the MultiPoint to WGS84 coordinate system (latitude/longitude).
        
        This property performs a coordinate transformation from the current CRS
        to WGS84 (EPSG:4326), which uses decimal degrees for latitude and longitude.
        This is particularly useful for:
        - Converting projected coordinates back to geographic coordinates
        - Preparing data for web mapping applications
        - Standardizing coordinates for global analysis

        Returns:
            MultiPoint: A new MultiPoint object with coordinates in WGS84 (lat/lon) system.
                        The original MultiPoint remains unchanged.
                        
        Example:
            >>> # MultiPoint in UTM coordinates
            >>> utm_points = MultiPoint(x=[500000, 600000], y=[4649776, 4749776], crs="EPSG:32633")
            >>> # Convert to lat/lon
            >>> latlon_points = utm_points.latlon
            >>> print(f"Latitudes: {latlon_points.y}")  # Latitude values
            >>> print(f"Longitudes: {latlon_points.x}")  # Longitude values
        """
        return self.to_crs(WGS84)

    @property
    def lat(self) -> np.ndarray:
        """
        Get the latitude coordinates of all points in WGS84 coordinate system.
        
        This property performs an on-the-fly transformation to WGS84 and extracts
        the y-coordinates (latitudes) in decimal degrees. Latitude values range
        from -90 (South Pole) to +90 (North Pole).

        Returns:
            np.ndarray: Array of latitude values in decimal degrees.
                        Returns empty array if geometry is empty.
                        
        Example:
            >>> # Points in any CRS
            >>> mp = MultiPoint(x=[500000, 600000], y=[4649776, 4749776], crs="EPSG:32633")
            >>> latitudes = mp.lat
            >>> print(f"Latitudes: {latitudes}")  # e.g., [41.9876, 42.8765]
        """
        # Transform to WGS84 on-the-fly and extract y-coordinates (latitudes)
        return self.latlon.y

    @property
    def lon(self) -> np.ndarray:
        """
        Get the longitude coordinates of all points in WGS84 coordinate system.
        
        This property performs an on-the-fly transformation to WGS84 and extracts
        the x-coordinates (longitudes) in decimal degrees. Longitude values range
        from -180 (antimeridian west) to +180 (antimeridian east).

        Returns:
            np.ndarray: Array of longitude values in decimal degrees.
                        Returns empty array if geometry is empty.
                        
        Example:
            >>> # Points in any CRS
            >>> mp = MultiPoint(x=[500000, 600000], y=[4649776, 4749776], crs="EPSG:32633")
            >>> longitudes = mp.lon
            >>> print(f"Longitudes: {longitudes}")  # e.g., [12.1234, 13.5678]
        """
        # Transform to WGS84 on-the-fly and extract x-coordinates (longitudes)
        return self.latlon.x
