from __future__ import annotations

from typing import Union, Iterable, TYPE_CHECKING

import shapely
import geopandas as gpd
from pyproj import Transformer

from .CRS import CRS, WGS84
from .spatial_geometry import SpatialGeometry

from .wrap_geometry import wrap_geometry

if TYPE_CHECKING:
    from .point import Point
    from .multi_point import MultiPoint

class VectorGeometry(SpatialGeometry):
    @property
    def wkt(self):
        return self.geometry.wkt
    
    def __repr__(self) -> str:
        return self.wkt

    def contain(self, other, crs: CRS = None, **kwargs) -> VectorGeometry:
        return self.__class__(other, crs=crs, **kwargs)

    def to_crs(self, crs: Union[CRS, str]) -> VectorGeometry:
        crs = CRS(crs)
        result = self.contain(
            shapely.ops.transform(Transformer.from_crs(self.crs, crs, always_xy=True).transform, self.geometry),
            crs=crs)
        return result

    @property
    def latlon(self):
        return self.to_crs(WGS84)

    @property
    def UTM(self):
        return self.to_crs(self.local_UTM_proj4)

    @property
    def mapping(self) -> dict:
        return shapely.geometry.mapping(self)

    def to_shapely(self):
        return shapely.geometry.shape(self.mapping)

    @property
    def shapely(self):
        return self.to_shapely()

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({}, geometry=[self.shapely], crs=self.crs.proj4)

    def to_geojson(self, filename: str):
        self.gdf.to_file(filename, driver="GeoJSON")

    @property
    def is_geographic(self) -> bool:
        return self.crs.is_geographic

    @property
    def projected(self):
        if self.is_geographic:
            return self.to_crs(self.local_UTM_proj4)
        else:
            return self

    def distances(self, points: Union[MultiPoint, Iterable[Point]]) -> gpd.GeoDataFrame:
        points = [wrap_geometry(point) for point in points]

        geometry = []
        distance_column = []

        for point in points:
            geometry.append(shapely.geometry.LineString([self, point]))
            distance_column.append(self.projected.distance(point.projected))

        distances = gpd.GeoDataFrame({"distance": distance_column}, geometry=geometry)

        return distances


class SingleVectorGeometry(VectorGeometry):
    pass

class MultiVectorGeometry(VectorGeometry):
    pass
