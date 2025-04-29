from __future__ import annotations

import warnings
from typing import List, Union

from .CRS import CRS, WGS84
from .point import Point
from .polygon import Polygon
from .spatial_geometry import SpatialGeometry


class BBox(SpatialGeometry):
    """
    A class representing a bounding box with a coordinate reference system (CRS).

    Attributes:
        x_min (float): The minimum x-coordinate.
        y_min (float): The minimum y-coordinate.
        x_max (float): The maximum x-coordinate.
        y_max (float): The maximum y-coordinate.
        crs (CRS): The coordinate reference system. Defaults to WGS84.
    """
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, crs: Union[CRS, str] = WGS84):
        """
        Initializes a new BBox object.

        Args:
            xmin (float): The minimum x-coordinate.
            ymin (float): The minimum y-coordinate.
            xmax (float): The maximum x-coordinate.
            ymax (float): The maximum y-coordinate.
            crs (Union[CRS, str]): The coordinate reference system. Defaults to WGS84.
        """
        super(BBox, self).__init__(crs=crs)

        self.x_min = xmin
        self.y_min = ymin
        self.x_max = xmax
        self.y_max = ymax

    def __repr__(self):
        """
        Returns a string representation of the BBox object.

        Returns:
            str: The string representation of the BBox object.
        """
        with warnings.catch_warnings():
            # Ignore warnings from CRS __repr__ if it has known issues
            warnings.simplefilter("ignore")
            return f'BBox(xmin={self.x_min}, ymin={self.y_min}, xmax={self.x_max}, ymax={self.y_max}, crs="{self.crs.__repr__()}")'

    def __iter__(self):
        """
        Allows iterating over the coordinates of the BBox.

        Yields:
            float: The next coordinate value (xmin, ymin, xmax, ymax).
        """
        for element in (self.x_min, self.y_min, self.x_max, self.y_max):
            yield element

    def __eq__(self, other: BBox) -> bool:
        """
        Checks if two BBox objects are equal.

        Args:
            other (BBox): The other BBox object to compare.

        Returns:
            bool: True if the BBoxes are equal, False otherwise.
        """
        return all([
            self.x_min == other.x_min,
            self.y_min == other.y_min,
            self.x_max == other.x_max,
            self.y_max == other.y_max,
            self.crs == other.crs
        ])

    @classmethod
    def merge(cls, bboxes: List[BBox], crs: CRS = None) -> BBox:
        """
        Merges a list of BBoxes into a single BBox.

        Args:
            bboxes (List[BBox]): The list of BBoxes to merge.
            crs (CRS, optional): The desired CRS for the merged BBox. 
                                 If None, the CRS of the first BBox in the list is used.

        Returns:
            BBox: The merged BBox.
        """
        if crs is None:
            crs = bboxes[0].crs

        # Ensure all bboxes are in the same CRS
        bboxes = [bbox.to_crs(crs) for bbox in bboxes]
        
        # Calculate the minimum and maximum coordinates
        xmin = min([bbox.x_min for bbox in bboxes])
        xmax = max([bbox.x_max for bbox in bboxes])
        ymin = min([bbox.y_min for bbox in bboxes])
        ymax = max([bbox.y_max for bbox in bboxes])  # Corrected ymax calculation to use max

        bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, crs=crs)

        return bbox

    @property
    def polygon(self) -> Polygon:
        """
        Returns a Polygon representation of the BBox.

        Returns:
            Polygon: The Polygon representation of the BBox.
        """
        return Polygon(
            [
                (self.x_min, self.y_max),
                (self.x_max, self.y_max),
                (self.x_max, self.y_min),
                (self.x_min, self.y_min)
            ],
            crs=self.crs
        )

    def transform(self, crs: Union[CRS, str]) -> BBox:
        """
        Transforms the BBox to a new CRS.

        Args:
            crs (Union[CRS, str]): The target CRS.

        Returns:
            BBox: The transformed BBox.
        """
        return self.polygon.to_crs(crs).bbox

    def to_crs(self, crs: Union[CRS, str]) -> BBox:
        """
        Alias for the `transform` method.

        Args:
            crs (Union[CRS, str]): The target CRS.

        Returns:
            BBox: The transformed BBox.
        """
        return self.transform(crs=crs)

    @property
    def latlon(self):
        """
        Returns the BBox in the WGS84 (latitude/longitude) CRS.

        Returns:
            BBox: The BBox in WGS84.
        """
        return self.transform(WGS84)

    @property
    def round(self) -> BBox:
        """
        Returns a new BBox with rounded coordinates.

        Returns:
            BBox: The BBox with rounded coordinates.
        """
        return BBox(
            xmin=round(self.x_min),
            ymin=round(self.y_min),
            xmax=round(self.x_max),
            ymax=round(self.y_max),
            crs=self.crs
        )

    @property
    def width(self) -> float:
        """
        Calculates the width of the BBox.

        Returns:
            float: The width of the BBox.
        """
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """
        Calculates the height of the BBox.

        Returns:
            float: The height of the BBox.
        """
        return self.y_max - self.y_min

    def buffer(self, buffer) -> BBox:
        """
        Creates a new BBox buffered by a given amount.

        Args:
            buffer (float): The buffer amount to apply.

        Returns:
            BBox: The buffered BBox.
        """
        return BBox(
            xmin=self.x_min - buffer,
            ymin=self.y_min - buffer,
            xmax=self.x_max + buffer,
            ymax=self.y_max + buffer,
            crs=self.crs
        )

    @property
    def centroid(self) -> Point:
        """
        Calculates the centroid of the BBox.

        Returns:
            Point: The centroid of the BBox.
        """
        return Point(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2,
            crs=self.crs
        )

    @property
    def bbox(self) -> BBox:
        """
        Returns the BBox itself.

        Returns:
            BBox: The BBox itself.
        """
        return self
