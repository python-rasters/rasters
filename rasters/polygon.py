from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np
import shapely

from .CRS import CRS, WGS84
from .vector_geometry import SingleVectorGeometry

if TYPE_CHECKING:
    from .bbox import BBox
    from .point import Point

class Polygon(SingleVectorGeometry):
    """
    Represents a polygon with a defined coordinate reference system (CRS).

    This class provides functionalities for creating, manipulating, and analyzing
    polygons using the `shapely` library. It inherits from `SingleVectorGeometry`
    and offers properties and methods for accessing geometric attributes such as
    centroid, exterior, bounds, and WKT representation.

    Args:
        *args: Variable arguments to initialize the polygon.
               - If the first argument is a `Polygon` instance, it copies the geometry and CRS.
               - Otherwise, it passes the arguments directly to `shapely.geometry.Polygon`.
        crs (Union[CRS, str], optional): The coordinate reference system of the polygon.
                                        Defaults to WGS84.

    Example:
        >>> from polygon import Polygon
        >>> polygon = Polygon([(0, 0), (1, 1), (1, 0)])  # Create a polygon from coordinates
        >>> print(polygon.centroid)  # Access the centroid of the polygon
        >>> print(polygon.wkt)  # Get the WKT representation of the polygon
    """
    def __init__(self, *args, crs: Union[CRS, str] = WGS84):
        if isinstance(args[0], Polygon):
            # Copy constructor - initialize from another Polygon instance
            geometry = args[0].geometry
            crs = args[0].crs
        else:
            # Initialize from coordinates or other arguments accepted by shapely.geometry.Polygon
            geometry = shapely.geometry.Polygon(*args)

        SingleVectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry

    @property
    def centroid(self) -> Point:
        """
        Returns the geometric center of the polygon.

        Returns:
            Point: The centroid of the polygon as a `Point` object.
        """
        from .point import Point
        return Point(self.geometry.centroid, crs=self.crs)

    @property
    def exterior(self):
        """
        Returns the exterior ring of the polygon.

        Returns:
            shapely.geometry.LinearRing: The exterior ring of the polygon.
        """
        return self.geometry.exterior

    @property
    def is_empty(self):
        """
        Checks if the polygon is empty.

        Returns:
            bool: True if the polygon is empty, False otherwise.
        """
        return self.geometry.is_empty

    @property
    def geom_type(self):
        """
        Returns the geometry type of the polygon.

        Returns:
            str: The geometry type as a string (e.g., 'Polygon').
        """
        return self.geometry.geom_type

    @property
    def bounds(self):
        """
        Returns the bounding box coordinates of the polygon.

        Returns:
            tuple: A tuple containing (minx, miny, maxx, maxy) coordinates.
        """
        return self.geometry.bounds

    @property
    def interiors(self):
        """
        Returns a list of interior rings of the polygon.

        Returns:
            list: A list of `shapely.geometry.LinearRing` objects representing the interior rings.
        """
        return self.geometry.interiors

    @property
    def wkt(self):
        """
        Returns the Well-Known Text (WKT) representation of the polygon.

        Returns:
            str: The WKT representation of the polygon.
        """
        return self.geometry.wkt

    @property
    def bbox(self) -> BBox:
        """
        Calculates and returns the bounding box of the polygon.

        The bounding box is computed based on the exterior ring of the polygon.

        Returns:
            BBox: The bounding box as a `BBox` object.
        """
        from .bbox import BBox
        x, y = self.exterior.xy
        x = np.array(x)
        y = np.array(y)
        x_min = float(np.nanmin(x))
        y_min = float(np.nanmin(y))
        x_max = float(np.nanmax(x))
        y_max = float(np.nanmax(y))
        bbox = BBox(x_min, y_min, x_max, y_max, crs=self.crs)

        return bbox
    