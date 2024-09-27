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
    def __init__(self, *args, crs: Union[CRS, str] = WGS84):
        if isinstance(args[0], Point):
            geometry = args[0].geometry
            crs = args[0].crs
        else:
            geometry = shapely.geometry.Point(*args)

        VectorGeometry.__init__(self, crs=crs)

        self.geometry = geometry

    @property
    def latlon(self) -> Point:
        return self.contain(
            gpd.GeoDataFrame({}, geometry=[self.geometry], crs=str(self.crs)).to_crs(str(WGS84)).geometry[0],
            crs=CRS(WGS84))

    @property
    def centroid(self) -> Point:
        return self

    @property
    def x(self):
        return self.geometry.x

    @property
    def y(self):
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
        return Polygon(
            shapely.geometry.Point.buffer(
                self.geometry,
                distance=distance,
                resolution=resolution,
                quadsegs=quadsegs,
                cap_style=cap_style,
                join_style=join_style,
                mitre_limit=mitre_limit,
                single_sided=single_sided
            ),
            crs=self.crs
        )
