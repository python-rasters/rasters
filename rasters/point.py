from __future__ import annotations

from typing import Union

import geopandas as gpd
import shapely
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE

from .constants import *
from .CRS import CRS, WGS84
from .vector_geometry import VectorGeometry, SingleVectorGeometry
from .polygon import Polygon

class Point(SingleVectorGeometry):
    """
    A class representing a Point geometry.

    Inherits from SingleVectorGeometry.

    Args:
        *args:  
            - If the first argument is a Point object, the geometry and crs will be extracted from it.  
            - Otherwise, the arguments are passed directly to shapely.geometry.Point constructor.
        crs (Union[CRS, str], optional): The coordinate reference system. Defaults to WGS84.

    Attributes:
        geometry (shapely.geometry.Point): The shapely geometry representing the point.
        crs (CRS): The coordinate reference system of the point.

    Example:
        >>> point = Point(10, 20, crs=WGS84)
        >>> print(point.x, point.y, point.crs)
        10 20 WGS84
    """
    def __init__(self, *args, crs: Union[CRS, str] = WGS84):
        if isinstance(args[0], Point):
            # If the first argument is a Point, extract geometry and crs
            geometry = args[0].geometry
            crs = args[0].crs
        else:
            # Otherwise, create a new shapely Point from the arguments
            geometry = shapely.geometry.Point(*args)

        # Initialize the base VectorGeometry class
        VectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry

    @property
    def latlon(self) -> Point:
        """
        Returns a new Point object with the coordinates transformed to WGS84 (latitude/longitude).

        Returns:
            Point: A new Point object with WGS84 coordinates.
        """
        # Convert the point to a GeoDataFrame for easy CRS transformation
        gdf = gpd.GeoDataFrame({}, geometry=[self.geometry], crs=str(self.crs))
        # Transform to WGS84 and extract the geometry
        transformed_geom = gdf.to_crs(str(WGS84)).geometry[0]
        # Create a new Point with the transformed geometry and WGS84 CRS
        return self.contain(transformed_geom, crs=CRS(WGS84))

    @property
    def centroid(self) -> Point:
        """
        Returns the centroid of the point, which is the point itself.

        Returns:
            Point: The point itself.
        """
        return self

    @property
    def x(self):
        """
        Returns the x-coordinate of the point.

        Returns:
            float: The x-coordinate.
        """
        return self.geometry.x

    @property
    def y(self):
        """
        Returns the y-coordinate of the point.

        Returns:
            float: The y-coordinate.
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
            single_sided=False) -> Polygon:
        """
        Creates a buffer polygon around the point.

        Args:
            distance: The buffer distance.
            resolution: The resolution of the buffer (number of segments used to approximate a circle).
            quadsegs: Use quadsegs to control the number of segments with which to approximate a quarter circle.
            cap_style: The cap style to use for the ends of linear rings.
            join_style: The join style to use for the corners of linear rings.
            mitre_limit: The mitre limit to use when creating the buffer.
            single_sided: Whether to create a single-sided buffer.

        Returns:
            Polygon: The buffer polygon.
        """
        # Create the buffer using shapely's buffer method
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
        # Return a new Polygon object with the buffer geometry and the point's CRS
        return Polygon(buffer_geom, crs=self.crs)
