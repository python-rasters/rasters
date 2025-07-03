from __future__ import annotations

from typing import Union

import geopandas as gpd
import shapely
from pyproj import Transformer

from .CRS import CRS, WGS84
from .spatial_geometry import SpatialGeometry

class VectorGeometry(SpatialGeometry):
    """
    Base class for representing vector geometries.

    Attributes:
        geometry (shapely.geometry.base.BaseGeometry): The shapely geometry object.
        crs (CRS): The coordinate reference system of the geometry.

    Properties:
        wkt (str): The Well-Known Text (WKT) representation of the geometry.
        latlon (VectorGeometry): The geometry projected to WGS84 (latitude/longitude).
        UTM (VectorGeometry): The geometry projected to the local UTM zone.
        mapping (dict): The GeoJSON-like mapping of the geometry.
        shapely (shapely.geometry.base.BaseGeometry): The shapely geometry object.
        gdf (gpd.GeoDataFrame): The geometry as a GeoDataFrame.
        is_geographic (bool): True if the geometry is in a geographic coordinate system.
        projected (VectorGeometry): The geometry in a projected coordinate system (UTM if geographic).
    """
    
    def __repr__(self) -> str:
        """
        Return the WKT representation of the geometry.
        """
        return self.wkt

    @property
    def wkt(self) -> str:
        """
        The Well-Known Text (WKT) representation of the geometry.
        """
        return self.geometry.wkt
    
    def contain(self, other, crs: CRS = None, **kwargs) -> VectorGeometry:
        """
        Contains another geometry within this geometry. This is useful for creating
        a new VectorGeometry object from a shapely geometry.

        Args:
            other: The geometry to contain.
            crs: The coordinate reference system of the geometry.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            VectorGeometry: The new VectorGeometry object.
        """
        return self.__class__(other, crs=crs, **kwargs)

    def to_crs(self, crs: Union[CRS, str]) -> VectorGeometry:
        """
        Project the geometry to a new coordinate reference system.

        Args:
            crs (Union[CRS, str]): The coordinate reference system to project to.

        Returns:
            VectorGeometry: The projected geometry.
        """
        crs = CRS(crs)
        result = self.contain(
            shapely.ops.transform(Transformer.from_crs(self.crs, crs, always_xy=True).transform, self.geometry),
            crs=crs)
        return result

    @property
    def latlon(self) -> VectorGeometry:
        """
        The geometry projected to WGS84 (latitude/longitude).
        """
        return self.to_crs(WGS84)

    @property
    def UTM(self) -> VectorGeometry:
        """
        The geometry projected to the local UTM zone.
        """
        return self.to_crs(self.local_UTM_proj4)

    @property
    def mapping(self) -> dict:
        """
        The GeoJSON-like mapping of the geometry.
        """
        return shapely.geometry.mapping(self)

    def to_shapely(self) -> shapely.geometry.base.BaseGeometry:
        """
        Convert the geometry to a shapely geometry object.
        """
        return shapely.geometry.shape(self.mapping)

    @property
    def shapely(self) -> shapely.geometry.base.BaseGeometry:
        """
        The shapely geometry object.
        """
        return self.to_shapely()

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """
        The geometry as a GeoDataFrame.
        """
        return gpd.GeoDataFrame({}, geometry=[self.shapely], crs=self.crs.proj4)

    def to_geojson(self, filename: str) -> None:
        """
        Save the geometry to a GeoJSON file.

        Args:
            filename (str): The name of the file to save to.
        """
        self.gdf.to_file(filename, driver="GeoJSON")

    @property
    def is_geographic(self) -> bool:
        """
        True if the geometry is in a geographic coordinate system.
        """
        return self.crs.is_geographic

    @property
    def projected(self) -> VectorGeometry:
        """
        The geometry in a projected coordinate system (UTM if geographic).
        """
        if self.is_geographic:
            return self.to_crs(self.local_UTM_proj4)
        else:
            return self


class SingleVectorGeometry(VectorGeometry):
    """
    Base class for single vector geometries (e.g., Point, LineString, Polygon).
    """
    pass

class MultiVectorGeometry(VectorGeometry):
    """
    Base class for multi vector geometries (e.g., MultiPoint, MultiLineString, MultiPolygon).
    """
    @property
    def geoms(self):
        return self.geometry.geoms

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.geoms[index]
        
        return self.geometry[index]

    # need to be able to iterate over geometries and apply self.contain to each one
    def __iter__(self):
        from .wrap_geometry import wrap_geometry

        for geometry in self.geoms:
            yield wrap_geometry(geometry, crs=self.crs)    

    # @property
    # def geoms(self) -> list[VectorGeometry]:
    #     """
    #     The list of geometries in the multi geometry.
    #     """ 
    #     return [self.contain(geometry) for geometry in self.geometry.geoms]
