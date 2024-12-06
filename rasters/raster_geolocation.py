from __future__ import annotations

from typing import Union, Tuple, Dict, TYPE_CHECKING

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

from .CRS import WGS84
from .raster_geometry import RasterGeometry
from .wrap_geometry import wrap_geometry

if TYPE_CHECKING:
    from .CRS import CRS
    from .point import Point
    from .polygon import Polygon
    from .raster_grid import RasterGrid


class RasterGeolocation(RasterGeometry):
    """
    This class encapsulates the geolocation of swath data using geolocation arrays.
    It inherits from RasterGeometry and provides methods for handling 
    geographic coordinates, indexing, and spatial operations.

    Attributes:
        _x (np.ndarray): Two-dimensional x-coordinate geolocation array.
        _y (np.ndarray): Two-dimensional y-coordinate geolocation array.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            crs: Union[CRS, str] = WGS84,
            **kwargs):
        """
        Initializes RasterGeolocation with x and y coordinate arrays and a CRS.

        Args:
            x (np.ndarray): Two-dimensional x-coordinate geolocation array.
            y (np.ndarray): Two-dimensional y-coordinate geolocation array.
            crs (Union[CRS, str], optional): CRS as proj4 string or pyproj.CRS object. 
                                             Defaults to WGS84.
            **kwargs: Keyword arguments passed to the parent class constructor.

        Raises:
            ValueError: If x or y coordinate arrays contain NaN values, 
                        or if no valid coordinates are provided.
        """
        super(RasterGeolocation, self).__init__(crs=crs, **kwargs)

        if np.any(np.isnan(x)):
            raise ValueError("x coordinate array contains NaN")

        if np.any(np.isnan(y)):
            raise ValueError("y coordinate array contains NaN")

        if not (np.isfinite(x).any() or np.isfinite(y).any()):
            raise ValueError("no valid coordinates given for coordinate field")

        self._x = x
        self._y = y

        if self.is_geographic:
            # Clip longitude values to -180 to 179.9999 and latitude values to -90 to 90
            self._x = np.clip(self._x, -180, 179.9999)
            self._y = np.clip(self._y, -90, 90)

    def __eq__(self, other: RasterGeolocation) -> bool:
        """
        Checks if this RasterGeolocation is equal to another.

        Args:
            other (RasterGeolocation): The other RasterGeolocation to compare to.

        Returns:
            bool: True if both objects have the same CRS and coordinate arrays, 
                  False otherwise.
        """
        return isinstance(other, RasterGeolocation) and \
               self.crs == other.crs and \
               np.array_equal(self.x, other.x) and \
               np.array_equal(self.y, other.y)

    def _slice(self, y_slice: slice, x_slice: slice) -> RasterGeometry:
        """
        Creates a subset of the RasterGeolocation using slices.

        Args:
            y_slice (slice): Slice object for the y-dimension.
            x_slice (slice): Slice object for the x-dimension.

        Returns:
            RasterGeometry: A new RasterGeolocation representing the subset.
        """
        crs = self.crs
        x = self.x[y_slice, x_slice]
        y = self.y[y_slice, x_slice]
        subset = RasterGeolocation(x=x, y=y, crs=crs)

        return subset

    @classmethod
    def from_vectors(
            cls,
            x_vector: np.ndarray,
            y_vector: np.ndarray,
            crs: Union[CRS, str] = WGS84) -> RasterGeolocation:
        """
        Creates a RasterGeolocation from x and y coordinate vectors.

        Args:
            x_vector (np.ndarray): One-dimensional x-coordinate vector.
            y_vector (np.ndarray): One-dimensional y-coordinate vector.
            crs (Union[CRS, str], optional): CRS as proj4 string or pyproj.CRS object. 
                                             Defaults to WGS84.

        Returns:
            RasterGeolocation: A new RasterGeolocation created from the vectors.
        """
        x, y = np.meshgrid(x_vector, y_vector)
        geolocation = RasterGeolocation(x, y, crs=crs)

        return geolocation

    def index_point(self, point: Point) -> Tuple[int, int]:
        """
        Finds the nearest neighbor index of a point in the geolocation arrays.

        Args:
            point (Point): The point to find the index for.

        Returns:
            Tuple[int, int]: The (row, column) index of the nearest neighbor.
        """
        dist, index = cKDTree(np.c_[self.x.ravel(), self.y.ravel()]).query((point.x, point.y))
        index = np.unravel_index(index, self.shape)

        return index

    def index(self, geometry: Union[RasterGeometry, Point, Polygon, Tuple[float, float, float, float]]):
        """
        Returns a boolean mask indexing the geolocation arrays based on a given geometry.

        Args:
            geometry (Union[RasterGeometry, Point, Polygon, Tuple[float, float, float, float]]): 
                The geometry to index with. Can be a RasterGeometry, Point, Polygon, or a 
                tuple representing a bounding box (xmin, ymin, xmax, ymax).

        Returns:
            np.ndarray: A boolean mask with the same shape as the geolocation arrays, 
                        where True values indicate the indexed locations.
        """
        geometry = wrap_geometry(geometry)
        xmin, ymin, xmax, ymax = geometry.bbox.transform(self.crs)

        index = np.logical_and(
            np.logical_and(
                self.x >= xmin,
                self.x <= xmax
            ),
            np.logical_and(
                self.y >= ymin,
                self.y <= ymax
            )
        )

        return index

    @property
    def x(self) -> np.ndarray:
        """np.ndarray: The two-dimensional x-coordinate geolocation array."""
        return self._x

    @property
    def y(self) -> np.ndarray:
        """np.ndarray: The two-dimensional y-coordinate geolocation array."""
        return self._y

    @property
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tuple[np.ndarray, np.ndarray]: A tuple containing the x and y coordinate arrays."""
        return self.x, self.y

    @property
    def rows(self) -> int:
        """int: The number of rows in the geolocation arrays."""
        return self.x.shape[0]

    @property
    def cols(self) -> int:
        """int: The number of columns in the geolocation arrays."""
        return self.y.shape[1]

    @property
    def x_min(self) -> float:
        """float: The minimum x-coordinate value."""
        return self.bbox.x_min

    @property
    def x_max(self) -> float:
        """float: The maximum x-coordinate value."""
        return self.bbox.x_max

    @property
    def y_min(self) -> float:
        """float: The minimum y-coordinate value."""
        return self.bbox.y_min

    @property
    def y_max(self) -> float:
        """float: The maximum y-coordinate value."""
        return self.bbox.y_max

    @property
    def width(self) -> float:
        """
        float: Width of extent in projected units. 
               Handles geographic coordinates crossing the antimeridian.
        """

        x_max = self.x_max
        x_min = self.x_min

        if self.is_geographic and self.crosses_antimeridian:
            if x_max < 0:
                x_max += 360

            if x_min < 0:
                x_min += 360

        return x_max - x_min

    @property
    def height(self) -> float:
        """
        Height of extent in projected units.
        """
        return self.y_max - self.y_min

    @property
    def cell_size(self) -> float:
        return float(min(np.nanmedian(self.row_distances), np.nanmedian(self.col_distances)))

    @property
    def cell_width(self) -> float:
        """
        Positive cell width in units of CRS.
        """
        # FIXME this assumes north orientation
        # return self.width / self.cols
        return self.cell_size

    @property
    def cell_height(self) -> float:
        """
        Negative cell height in units of CRS.
        """
        # FIXME this assumes north orientation
        # return -1.0 * (self.height / self.rows)
        return -1 * self.cell_size

    @property
    def boundary(self) -> Polygon:
        from .polygon import Polygon

        y_indices, x_indices = self.boundary_indices
        x_boundary = self.x[y_indices, x_indices]
        y_boundary = self.y[y_indices, x_indices]
        points = np.c_[x_boundary, y_boundary]
        boundary = Polygon(points, crs=self.crs)

        return boundary

    @property
    def corner_polygon(self) -> Polygon:
        """
        Draw polygon through the corner coordinates of geolocation arrays.
        :return: shapely.geometry.Polygon of corner coordinate boundary
        """
        from .polygon import Polygon

        return Polygon([
            (self.x[0, 0], self.y[0, 0]),
            (self.x[0, self.x.shape[1] - 1], self.y[0, self.y.shape[1] - 1]),
            (self.x[self.x.shape[0] - 1, self.x.shape[1] - 1], self.y[self.y.shape[0] - 1, self.y.shape[1] - 1]),
            (self.x[self.x.shape[0] - 1, 0], self.y[self.y.shape[0] - 1, 0])
        ])

    def resize(self, dimensions: Tuple[int, int], order: int = 2) -> RasterGeometry:
        if len(dimensions) != 2:
            raise ValueError("coordinate field dimensionality must be two-dimensional")

        rows_target, cols_target = dimensions

        if not isinstance(rows_target, int) or not isinstance(cols_target, int):
            raise ValueError(f"dimensions must be two-tuple of ints, not ({type(rows_target)}, {type(cols_target)})")

        zoom_factors = (float(rows_target) / float(self.rows), float(cols_target) / float(self.cols))
        use_shift = self.is_geographic and self.crosses_antimeridian

        x = self._x
        y = self._y

        if use_shift:
            x = np.where(x < 0, 360.0 + x, x)

        x = zoom(x, zoom_factors, order=order)
        y = zoom(y, zoom_factors, order=order)

        if use_shift:
            x = np.where(x >= 180.0, x - 360.0, x)

        resized_field = RasterGeolocation(x, y, crs=self.crs)

        return resized_field

    @property
    def grid(self) -> RasterGrid:
        return self.generate_grid(dest_crs=self.crs)

    def to_dict(self, output_dict: Dict = None, write_geolocation_arrays: bool = False) -> Dict:
        # FIXME this should conform to the CoverageJSON standard
        if output_dict is None:
            output_dict = {}

        output_dict['type'] = 'swath'
        output_dict['crs'] = self.proj4
        output_dict['x'] = self.x
        output_dict['y'] = self.y
        output_dict['rows'] = self.rows
        output_dict['cols'] = self.cols

        if write_geolocation_arrays:
            lat, lon = self.latlon_matrices
            output_dict['latitude'] = lat
            output_dict['longitude'] = lon

        return output_dict
