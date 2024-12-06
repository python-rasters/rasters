from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any, Optional, Union

import pyproj


class CRS(pyproj.CRS):
    """
    A class representing a Coordinate Reference System (CRS), extending pyproj.CRS.

    This class provides additional functionalities and representations for CRS objects,
    including EPSG code handling, proj4 string representation, and conversion to different formats.

    Attributes:
        projparams (Any): Projection parameters used to initialize the CRS object.
    """

    def __init__(self, projparams: Any = None, **kwargs) -> None:
        """
        Initializes a new CRS object.

        Args:
            projparams (Any): Projection parameters used to initialize the CRS object.
            **kwargs: Additional keyword arguments passed to the pyproj.CRS constructor.
        """
        super(CRS, self).__init__(projparams=projparams, **kwargs)
        self._proj4_string = None  # Cache for proj4 string

    def __repr__(self) -> str:
        """
        Returns a string representation of the CRS object.

        If the CRS has an EPSG code, it returns the EPSG string (e.g., "EPSG:4326").
        Otherwise, it returns the proj4 string representation.

        Returns:
            str: The string representation of the CRS object.
        """
        epsg_string = self.epsg_string

        if epsg_string is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.to_proj4()
        else:
            return epsg_string

    def __eq__(self, other: Union[CRS, str]) -> bool:
        """
        Checks if two CRS objects are equal.

        If both CRS objects have EPSG codes, it compares the EPSG codes.
        Otherwise, it uses the default equality check from pyproj.CRS.

        Args:
            other (Union[CRS, str]): The other CRS object or string representation to compare.

        Returns:
            bool: True if the CRS objects are equal, False otherwise.
        """
        if isinstance(other, CRS) and self.to_epsg() is not None and other.to_epsg() is not None:
            return self.to_epsg() == other.to_epsg()

        return super(CRS, self).__eq__(other=other)

    # @property
    # def _crs(self) -> Any:
    #     """
    #     Retrieve the underlying Cython based _CRS object. This is used internally by pyproj
    #     for performance optimization.

    #     Note: The implementation might change in future pyproj versions.
    #     """
    #     return self._crs  # Directly access the _crs attribute

    @property
    def epsg_string(self) -> Optional[str]:
        """
        Returns the EPSG string representation of the CRS if it has an EPSG code.

        Returns:
            Optional[str]: The EPSG string (e.g., "EPSG:4326") or None if no EPSG code is found.
        """
        epsg_code = self.to_epsg()

        if epsg_code is None:
            return None

        epsg_string = f"EPSG:{epsg_code}"

        return epsg_string

    @property
    def epsg_url(self) -> Optional[str]:
        """
        Returns the URL of the EPSG registry entry for the CRS if it has an EPSG code.

        Returns:
            Optional[str]: The EPSG URL or None if no EPSG code is found.
        """
        epsg_code = self.to_epsg()

        if epsg_code is None:
            return None

        # You might need to verify this URL
        epsg_url = f"https://epsg.org/{epsg_code}"

        return epsg_url

    @property
    def coverage(self) -> OrderedDict:
        """
        Returns an OrderedDict representing the coverage of the CRS.

        This includes the coordinate system type (GeographicCRS or ProjectedCRS) and
        the EPSG URL if available.

        Returns:
            OrderedDict: The coverage information.
        """
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

        try:
            area_of_use = self.area_of_use
            coverage["area_of_use"] = {
                "name": area_of_use.name,
                "bounds": area_of_use.bounds,
            }
        except Exception:  # Catch potential exceptions from pyproj
            pass

        coverage["system"] = system

        return coverage

    def to_pyproj(self) -> pyproj.CRS:
        """
        Converts the CRS to a pyproj.CRS object.

        Returns:
            pyproj.CRS: The pyproj.CRS object.
        """
        return pyproj.CRS(self.to_wkt())

    @property
    def proj4(self) -> str:
        """
        Returns the proj4 string representation of the CRS.

        Returns:
            str: The proj4 string.
        """
        if self._proj4_string is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._proj4_string = self.to_proj4()
        return self._proj4_string

WGS84 = CRS("EPSG:4326")
