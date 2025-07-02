from __future__ import annotations

from typing import Union, Iterable, TYPE_CHECKING

import geopandas as gpd
import shapely
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE

from rasters.wrap_geometry import wrap_geometry
from .CRS import CRS, WGS84
from .vector_geometry import VectorGeometry, SingleVectorGeometry

# TYPE_CHECKING imports to avoid circular imports at runtime
if TYPE_CHECKING:
    from .multi_point import MultiPoint
    from .bbox import BBox
    from .polygon import Polygon


class Point(SingleVectorGeometry):
    """
    A class representing a Point geometry with coordinate reference system support.

    This class provides a wrapper around shapely.geometry.Point with additional
    functionality for coordinate reference system (CRS) handling, transformations,
    and geometric operations specific to geospatial applications.

    Inherits from SingleVectorGeometry, which provides base functionality for
    single geometry objects with CRS support.

    Args:
        *args: Variable arguments for Point construction:
            - If the first argument is a Point object, the geometry and crs will be extracted from it.  
            - Otherwise, the arguments are passed directly to shapely.geometry.Point constructor.
        crs (Union[CRS, str], optional): The coordinate reference system. Defaults to WGS84.
        x (float, optional): X-coordinate when using keyword arguments for construction.
        y (float, optional): Y-coordinate when using keyword arguments for construction.

    Attributes:
        geometry (shapely.geometry.Point): The underlying shapely geometry representing the point.
        crs (CRS): The coordinate reference system of the point.

    Example:
        >>> # Create a point using positional arguments
        >>> point = Point(10, 20, crs=WGS84)
        >>> print(point.x, point.y, point.crs)
        10 20 WGS84
        
        >>> # Create a point using keyword arguments
        >>> point = Point(x=10, y=20, crs=WGS84)
        
        >>> # Create a point from another point
        >>> point2 = Point(point)
    """
    def __init__(self, *args, crs: Union[CRS, str] = WGS84, x: float = None, y: float = None):
        """
        Initialize a Point geometry with coordinate reference system support.
        
        Supports multiple construction patterns:
        1. From another Point object (copies geometry and CRS)
        2. From x, y keyword arguments
        3. From positional arguments passed to shapely.geometry.Point
        """
        if len(args) > 0 and isinstance(args[0], Point):
            # Copy constructor: extract geometry and CRS from existing Point
            geometry = args[0].geometry
            crs = args[0].crs
        elif x is not None and y is not None:
            # Keyword argument constructor: create Point from x, y coordinates
            geometry = shapely.geometry.Point(x, y)
        else:
            # Positional argument constructor: pass arguments directly to shapely
            geometry = shapely.geometry.Point(*args)
    
        # Initialize the parent VectorGeometry class with the specified CRS
        VectorGeometry.__init__(self, crs=crs)
    
        # Store the shapely geometry object
        self.geometry = geometry

    @property
    def latlon(self) -> Point:
        """
        Transform the point to WGS84 coordinate system (latitude/longitude).
        
        This property performs a coordinate transformation from the current CRS
        to WGS84 (EPSG:4326), which uses decimal degrees for latitude and longitude.
        Useful for converting projected coordinates back to geographic coordinates.

        Returns:
            Point: A new Point object with coordinates in WGS84 (lat/lon) system.
            
        Example:
            >>> # Point in UTM coordinates
            >>> utm_point = Point(500000, 4649776, crs="EPSG:32633")
            >>> # Convert to lat/lon
            >>> latlon_point = utm_point.latlon
            >>> print(f"Lat: {latlon_point.y}, Lon: {latlon_point.x}")
        """
        # Create a temporary GeoDataFrame for CRS transformation
        # This leverages GeoPandas' efficient coordinate transformation capabilities
        gdf = gpd.GeoDataFrame({}, geometry=[self.geometry], crs=str(self.crs))
        
        # Transform to WGS84 and extract the transformed geometry
        transformed_geom = gdf.to_crs(str(WGS84)).geometry[0]
        
        # Create and return a new Point with the transformed geometry in WGS84
        return self.contain(transformed_geom, crs=CRS(WGS84))

    @property
    def centroid(self) -> Point:
        """
        Return the centroid of the point geometry.
        
        For a Point geometry, the centroid is the point itself, as it represents
        a single location with no area or extent. This property is provided for
        consistency with other geometry types that have meaningful centroids.

        Returns:
            Point: The point itself (self).
        """
        return self

    @property
    def x(self) -> float:
        """
        Get the x-coordinate (longitude or easting) of the point.
        
        The interpretation of the x-coordinate depends on the coordinate reference system:
        - In geographic CRS (like WGS84): longitude in decimal degrees
        - In projected CRS (like UTM): easting in meters

        Returns:
            float: The x-coordinate value.
        """
        return self.geometry.x

    @property
    def y(self) -> float:
        """
        Get the y-coordinate (latitude or northing) of the point.
        
        The interpretation of the y-coordinate depends on the coordinate reference system:
        - In geographic CRS (like WGS84): latitude in decimal degrees  
        - In projected CRS (like UTM): northing in meters

        Returns:
            float: The y-coordinate value.
        """
        return self.geometry.y

    def buffer(
            self,
            distance,
            resolution=16,
            quadsegs=None,
            cap_style=CAP_STYLE.round,
            join_style=JOIN_STYLE.round,
            mitre_limit=5.0,
            single_sided=False) -> "Polygon":
        """
        Create a buffer polygon around the point.
        
        A buffer operation creates a polygon that represents all points within
        a specified distance of the original point. For a point, this typically
        creates a circular or nearly circular polygon centered on the point.

        Args:
            distance (float): The buffer distance in the units of the point's CRS.
                             For geographic CRS, this is typically degrees.
                             For projected CRS, this is typically meters.
            resolution (int, optional): Number of segments used to approximate curved sections.
                                      Higher values create smoother curves. Defaults to 16.
            quadsegs (int, optional): Number of segments per quarter circle. 
                                    If provided, overrides resolution. Defaults to None.
            cap_style (CAP_STYLE, optional): Style for line endings. Defaults to round.
            join_style (JOIN_STYLE, optional): Style for line joins. Defaults to round.
            mitre_limit (float, optional): Limit for mitre joins. Defaults to 5.0.
            single_sided (bool, optional): Whether to create single-sided buffer. 
                                         Defaults to False.

        Returns:
            Polygon: A polygon representing the buffered area around the point.
            
        Example:
            >>> point = Point(0, 0, crs=WGS84)
            >>> # Create a 1000 meter buffer (assuming projected CRS)
            >>> buffer_poly = point.buffer(1000)
        """
        # Import here to avoid circular imports
        from .polygon import Polygon

        # Create the buffer geometry using shapely's buffer method with all parameters
        buffer_geom = shapely.geometry.Point.buffer(
            self.geometry,
            distance=distance,
            resolution=resolution,
            quadsegs=quadsegs,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided
        )
        
        # Return a new Polygon object with the buffer geometry and the same CRS as the point
        return Polygon(buffer_geom, crs=self.crs)

    @property
    def bbox(self) -> "BBox":
        """
        Get the bounding box (envelope) of the point.
        
        For a point geometry, the bounding box is a degenerate rectangle where
        all four corners have the same coordinates as the point. This creates
        a "zero-area" bounding box useful for spatial indexing and operations.
    
        Returns:
            BBox: A bounding box where min and max coordinates are identical to the point.
                  
        Example:
            >>> point = Point(10, 20, crs=WGS84)
            >>> bbox = point.bbox
            >>> # bbox represents the rectangle (10, 20, 10, 20)
        """
        # Import here to avoid circular imports
        from .bbox import BBox
        
        # Create a bounding box where min and max coordinates are the same (the point itself)
        return BBox(self.x, self.y, self.x, self.y, crs=self.crs)
    
    def distance(self, other: Point) -> float:
        """
        Calculate the Euclidean distance between this point and another point.
        
        The calculation is performed in a projected coordinate system to ensure
        accurate distance measurements. Both points are projected to the same
        coordinate system before calculating the distance using the Pythagorean theorem.

        Args:
            other (Point): The other point to calculate distance to. Can also accept
                          a shapely.geometry.Point (will be wrapped with WGS84 CRS).

        Returns:
            float: The distance between the two points in the units of the projected CRS
                   (typically meters for most projected coordinate systems).
                   
        Example:
            >>> point1 = Point(0, 0, crs=WGS84)
            >>> point2 = Point(0, 1, crs=WGS84)  
            >>> distance = point1.distance(point2)
            >>> # Returns distance in meters after projection
        """
        # Handle case where other is a raw shapely Point (wrap it with WGS84 CRS)
        if isinstance(other, shapely.geometry.Point):
            other = Point(other, crs=WGS84)

        # Project this point to an appropriate projected coordinate system
        projected = self.projected
        
        # Project the other point to the same coordinate system for accurate distance calculation
        other_projected = other.to_crs(projected.crs)
        other_projected._crs = projected.crs
        
        # Extract coordinates from both projected points
        x, y = projected.geometry.x, projected.geometry.y
        other_x, other_y = other_projected.geometry.x, other_projected.geometry.y
        
        # Calculate Euclidean distance using the Pythagorean theorem
        distance = ((x - other_x) ** 2 + (y - other_y) ** 2) ** 0.5

        return distance

    def distances(self, points: Union["MultiPoint", Iterable["Point"]]) -> gpd.GeoDataFrame:
        """
        Calculate distances between this point and a collection of other points.
        
        This method computes the distance from this point to each point in a collection,
        returning both the numerical distances and LineString geometries connecting
        this point to each target point. This is useful for proximity analysis
        and visualization of spatial relationships.

        Args:
            points (Union[MultiPoint, Iterable[Point]]): A collection of points to calculate 
                                                        distances to. Can be a MultiPoint geometry
                                                        or any iterable of Point objects.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing:
                - 'distance' column: Numerical distances to each point
                - 'geometry' column: LineString geometries connecting this point to each target point
                The CRS of the resulting GeoDataFrame matches the input points.
                
        Example:
            >>> origin = Point(0, 0, crs=WGS84)
            >>> target_points = [Point(1, 0), Point(0, 1), Point(1, 1)]
            >>> distances_gdf = origin.distances(target_points)
            >>> print(distances_gdf['distance'].values)  # Array of distances
        """
        # Wrap the input to ensure it's a proper geometry object with CRS
        points = wrap_geometry(points)

        # Initialize lists to store results
        geometry = []        # Will contain LineString geometries
        distance_column = [] # Will contain numerical distance values

        # Process each point in the collection
        for point in [wrap_geometry(point) for point in points.geoms]:
            # Ensure the point has the same CRS as the collection
            point._crs = points._crs
            
            # Create a LineString geometry connecting this point to the target point
            geometry.append(shapely.geometry.LineString([self.geometry, point.geometry]))
            
            # Calculate and store the numerical distance
            distance_column.append(self.distance(point))

        # Create and return a GeoDataFrame with distances and connecting lines
        distances = gpd.GeoDataFrame({"distance": distance_column}, geometry=geometry)

        return distances