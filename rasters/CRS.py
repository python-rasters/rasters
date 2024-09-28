from __future__ import annotations

from typing import Any, Union

import warnings
import pyproj

class CRS(pyproj.CRS):
    def __init__(self, projparams: Any = None, **kwargs) -> None:
        super(CRS, self).__init__(projparams=projparams, **kwargs)

    def __repr__(self) -> str:
        epsg_string = self.epsg_string

        if epsg_string is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.to_proj4()
        else:
            return epsg_string

    def __eq__(self, other: Union[CRS, str]) -> bool:
        if self.to_epsg() is not None and other.to_epsg() is not None:
            return self.to_epsg() == other.to_epsg()

        return super(CRS, self).__eq__(other=other)

    @property
    def _crs(self):
        """
        Retrieve the Cython based _CRS object for this thread.
        """
        if not hasattr(self, "_local") or self._local is None:
            from pyproj.crs.crs import CRSLocal
            self._local = CRSLocal()

        if self._local.crs is None:
            self._local.crs = _CRS(self.srs)
        return self._local.crs

    # factories

    @classmethod
    def center_aeqd(cls, center_coord: Point) -> CRS:
        """
        Generate Azimuthal Equal Area CRS centered at given lat/lon.
        :param center_coord: shapely.geometry.Point object containing latitute and longitude point of center of CRS
        :return: pyproj.CRS object of centered CRS
        """
        return CRS(f"+proj=aeqd +lat_0={center_coord.y} +lon_0={center_coord.x}")

    @classmethod
    def local_UTM_proj4(cls, point_latlon: Union[Point, str]) -> CRS:
        if isinstance(point_latlon, str):
            point_latlon = shapely.wkt.loads(point_latlon)

        lat = point_latlon.y
        lon = point_latlon.x
        UTM_zone = (math.floor((lon + 180) / 6) % 60) + 1
        UTM_proj4 = CRS(
            f"+proj=utm +zone={UTM_zone} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs")

        return UTM_proj4

    # properties

    @property
    def epsg_string(self) -> Optional[str]:
        epsg_code = self.to_epsg()

        if epsg_code is None:
            return None

        epsg_string = f"EPSG:{epsg_code}"

        return epsg_string

    @property
    def epsg_url(self) -> Optional[str]:
        epsg_code = self.to_epsg()

        if epsg_code is None:
            return None

        epsg_url = f"http://www.opengis.net/def/crs/EPSG/9.9.1/{epsg_code}"

        return epsg_url

    @property
    def coverage(self) -> OrderedDict:
        coverage = OrderedDict()
        coverage["coordinates"] = ["x", "y"]
        system = OrderedDict()

        if self.is_geographic:
            system["type"] = "GeographicCRS"
        else:
            system["type"] = "ProjectedCRS"

        epsg_url = self.epsg_url

        if epsg_url is not None:
            system["id"] = epsg_url

        coverage["system"] = system

        return coverage

    def to_pyproj(self) -> pyproj.CRS:
        return pyproj.CRS(self.to_wkt())

    @property
    def pyproj(self) -> pyproj.CRS:
        return self.to_pyproj()

    @property
    def proj4(self) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.to_proj4()


WGS84 = CRS("EPSG:4326")