from __future__ import annotations

from typing import List, Union
import warnings

from .CRS import CRS, WGS84
from .spatial_geometry import SpatialGeometry
from .polygon import Polygon

class BBox(SpatialGeometry):
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, crs: Union[CRS, str] = WGS84):
        super(BBox, self).__init__(crs=crs)

        self.x_min = xmin
        self.y_min = ymin
        self.x_max = xmax
        self.y_max = ymax

    def __repr__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f'BBox(xmin={self.x_min}, ymin={self.y_min}, xmax={self.x_max}, ymax={self.y_max}, crs="{self.crs.__repr__()}")'

    def __iter__(self):
        for element in (self.x_min, self.y_min, self.x_max, self.y_max):
            yield element

    def __eq__(self, other: BBox) -> bool:
        return all([
            self.x_min == other.x_min,
            self.y_min == other.y_min,
            self.x_max == other.x_max,
            self.y_max == other.y_max,
            self.crs == other.crs
        ])

    @classmethod
    def merge(cls, bboxes: List[BBox], crs: CRS = None) -> BBox:
        if crs is None:
            crs = bboxes[0].crs

        bboxes = [bbox.to_crs(crs) for bbox in bboxes]
        xmin = min([bbox.x_min for bbox in bboxes])
        xmax = max([bbox.x_max for bbox in bboxes])
        ymin = min([bbox.y_min for bbox in bboxes])
        ymax = min([bbox.y_max for bbox in bboxes])
        bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, crs=crs)

        return bbox

    @property
    def polygon(self) -> Polygon:
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
        return self.polygon.to_crs(crs).bbox

    def to_crs(self, crs: Union[CRS, str]) -> BBox:
        return self.transform(crs=crs)

    @property
    def latlon(self):
        return self.transform(WGS84)

    @property
    def round(self) -> BBox:
        return BBox(
            xmin=round(self.x_min),
            ymin=round(self.y_min),
            xmax=round(self.x_max),
            ymax=round(self.y_max),
            crs=self.crs
        )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def buffer(self, buffer) -> BBox:
        return BBox(
            xmin=self.x_min - buffer,
            ymin=self.y_min - buffer,
            xmax=self.x_max + buffer,
            ymax=self.y_max + buffer,
            crs=self.crs
        )

    @property
    def bbox(self) -> BBox:
        return self
