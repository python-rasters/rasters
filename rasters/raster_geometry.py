from __future__ import annotations

import json
import logging
import warnings
from abc import abstractmethod
from typing import Union, List, Tuple, Dict, TYPE_CHECKING

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.ndimage import shift
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from .CRS import WGS84
from .spatial_geometry import SpatialGeometry
from .wrap_geometry import wrap_geometry

if TYPE_CHECKING:
    from .CRS import CRS
    from .bbox import BBox
    from .spatial_geometry import SpatialGeometry
    from .point import Point
    from .multi_point import MultiPoint
    from .polygon import Polygon
    from .multi_polygon import MultiPolygon
    from .kdtree import KDTree
    from .raster_geolocation import RasterGeolocation
    from .raster_grid import RasterGrid


class RasterGeometry(SpatialGeometry):
    """
    This is the base class for encapsulating a raster's geography.
    Child classes will implement behaviour for the coordinate field of a swath and transform of a grid.
    """
    logger = logging.getLogger(__name__)

    geometry_type = None

    def __init__(self, crs: Union[CRS, str] = WGS84, **kwargs):
        """
        :param crs: CRS as proj4 string or pyproj.CRS object
        """
        from .CRS import CRS

        if isinstance(crs, CRS):
            self._crs = crs
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._crs = CRS(crs)

        # create empty geographic geolocation arrays
        self._lat = None
        self._lon = None
        self._cell_size_meters = None
        self._bbox = None
        self._crosses_antimeridian = None

    def __repr__(self):
        x_min, y_min, x_max, y_max = self.bbox

        display_dict = {
            "dimensions": {
                "rows": int(self.rows),
                "cols": int(self.cols)
            },
            "bbox": {
                "xmin": float(x_min),
                "ymin": float(y_min),
                "xmax": float(x_max),
                "ymax": float(y_max)
            },
            "crs": str(self.crs.__repr__()),
            "resolution": {
                "cell_width": float(self.cell_width),
                "cell_height": float(self.cell_height)
            }
        }

        display_string = json.dumps(display_dict, indent=2)
        # display_string = yaml.dump(display_dict)

        return display_string

    @abstractmethod
    def __eq__(self, other: RasterGeometry) -> bool:
        pass

    def __getitem__(self, key: Union[slice, int, tuple]):
        if isinstance(key, (slice, int, tuple)):
            return self._key(key)

        else:
            raise KeyError('unrecognized key')

    @abstractmethod
    def _subset_index(self, y_slice: slice, x_slice: slice):
        pass

    @abstractmethod
    def _slice_coords(self, y_slice: slice, x_slice: slice):
        pass

    def _slice(self, y_slice: slice, x_slice: slice) -> RasterGeometry:
        from .point import Point

        if any([any([isinstance(index, float) for index in (slice.start, slice.stop, slice.step)]) for slice in
                (y_slice, x_slice)]):
            return self._slice_coords(y_slice, x_slice)

        y_start, y_end, y_step = y_slice.indices(self.rows)
        x_start, x_end, x_step = x_slice.indices(self.cols)

        if y_end - y_start == 1 and x_end - x_start == 1:
            return Point(self.x[y_start, x_start], self.y[y_start, x_start], crs=self.crs)

        if y_step != 1 or x_step != 1:
            raise NotImplementedError('stepped slicing is not implemented')

        return self._subset_index(slice(y_start, y_end, None), slice(x_start, x_end))

    def _key(self, key: Union[int, tuple, Ellipsis]) -> RasterGeometry:

        if key is Ellipsis:
            return self

        if isinstance(key, int):
            key = (key,)
        elif not isinstance(key, tuple):
            raise KeyError(f"invalid key: {key}")

        slices = []

        for i in range(2):
            if i > len(key) - 1:
                slices.append(slice(None, None, None))
            elif isinstance(key[i], int):
                slices.append(slice(key[i], key[i] + 1, None))
            elif isinstance(key[i], slice):
                slices.append(key[i])
            else:
                raise KeyError(f"invalid index: {key[i]}")

        y_slice, x_slice = slices

        return self._slice(y_slice, x_slice)

    @property
    def _matplotlib_extent(self) -> List[float]:
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    @property
    def boundary_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        height = self.rows
        width = self.cols

        # Handle empty grids
        if height == 0 or width == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        
        # Handle single cell grid
        if height == 1 and width == 1:
            return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)
        
        y_indices = []
        x_indices = []
        
        # Top boundary (first row)
        y_top = np.zeros(width, dtype=np.int32)
        x_top = np.arange(width, dtype=np.int32)
        y_indices.append(y_top)
        x_indices.append(x_top)
        
        if height > 1:
            # Right boundary (last column, excluding the first cell to avoid duplication)
            y_right = np.arange(1, height, dtype=np.int32)
            x_right = np.full_like(y_right, width - 1, dtype=np.int32)
            y_indices.append(y_right)
            x_indices.append(x_right)
            
            if width > 1:
                # Bottom boundary (last row, excluding the last cell to avoid duplication)
                y_bottom = np.full(width - 1, height - 1, dtype=np.int32)
                x_bottom = np.arange(width - 2, -1, -1, dtype=np.int32)
                y_indices.append(y_bottom)
                x_indices.append(x_bottom)
            
            if width > 1 and height > 2:
                # Left boundary (first column, excluding the first and last cells to avoid duplication)
                y_left = np.arange(height - 2, 0, -1, dtype=np.int32)
                x_left = np.zeros_like(y_left, dtype=np.int32)
                y_indices.append(y_left)
                x_indices.append(x_left)
        
        # Concatenate all boundary indices
        y_indices = np.concatenate(y_indices) if y_indices else np.array([], dtype=np.int32)
        x_indices = np.concatenate(x_indices) if x_indices else np.array([], dtype=np.int32)
        
        return y_indices, x_indices

    @property
    @abstractmethod
    def grid(self) -> RasterGrid:
        pass

    @property
    def is_geographic(self) -> bool:
        return self.crs.is_geographic

    @property
    def proj4(self) -> str:
        """
        CRSection as proj4 string
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.crs.to_proj4()

    @property
    def local_UTM_EPSG(self) -> str:
        centroid = self.centroid_latlon
        lat = centroid.y
        lon = centroid.x
        EPSG_code = int(f"{326 if lat >= 0 else 327}{((np.floor((lon + 180) / 6) % 60) + 1):02d}")

        return EPSG_code

    @property
    def local_UTM_proj4(self) -> str:
        centroid = self.centroid.latlon
        lat = centroid.y
        lon = centroid.x
        UTM_zone = (np.floor((lon + 180) / 6) % 60) + 1
        UTM_proj4 = f"+proj=utm +zone={UTM_zone} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs"

        return UTM_proj4

    @property
    def is_point(self) -> bool:
        """
        Raster reduces to a point if all dimensions are a single value in length
        """
        return all([dim == 1 for dim in self.shape])

    @property
    def point(self) -> Point:
        from .point import Point

        if not self.is_point:
            return ValueError("not a single point")

        x = self.x[0, 0]
        y = self.y[0, 0]

        point = Point(x, y, crs=self.crs)

        return point

    @abstractmethod
    def index_point(self, point: Point) -> Tuple[int, int]:
        pass

    @abstractmethod
    def index(self, geometry: Union[RasterGeometry, Point, Polygon, Tuple[float, float, float, float]]):
        pass

    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """
        Geolocation array of x-coordinates.
        """
        pass

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        """
        Geolocation array of y-coordinates.
        """
        pass

    @property
    @abstractmethod
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    def latlon_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lat is None or self._lon is None:
            if self.is_geographic:
                self._lon, self._lat = self.xy
            else:
                x = self.x
                y = self.y
                lat, lon = Transformer.from_crs(self.crs, WGS84).transform(x, y)
                self._lon = lon
                self._lat = lat

        warnings.filterwarnings('ignore')
        self._lat = np.where(self._lat < -90, np.nan, self._lat)
        self._lat = np.where(self._lat > 90, np.nan, self._lat)
        self._lon = np.where(self._lon < -180, np.nan, self._lon)
        self._lon = np.where(self._lon > 180, np.nan, self._lon)
        warnings.resetwarnings()

        return self._lat, self._lon

    @property
    def lat(self) -> np.ndarray:
        """
        Geolocation array of latitude.
        """
        lat, lon = self.latlon_matrices

        if np.all(np.isnan(lat)):
            raise ValueError("blank latitude array generated")

        return lat

    @property
    def lon(self) -> np.ndarray:
        """
        Geolocation array of longitude.
        """
        lat, lon = self.latlon_matrices

        if np.all(np.isnan(lon)):
            raise ValueError("blank longitude array generated")

        return lon

    @abstractmethod
    def rows(self) -> int:
        """
        Y-dimension of surface in rows.
        """
        pass

    @abstractmethod
    def cols(self) -> int:
        """
        X-dimension of surface in columns.
        """
        pass

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Dimensions of surface in rows and columns.
        """
        return self.rows, self.cols

    @abstractmethod
    def resize(self, dimensions: Tuple[int, int]) -> RasterGeometry:
        """
        Reshape dimensions of surface to target (rows, cols) tuple.
        """
        pass

    def zoom(self, factor: float) -> RasterGeometry:
        return self.resize((int(self.rows * factor), int(self.cols * factor)))

    @abstractmethod
    def width(self) -> float:
        """
        Width of extent in projected units.
        """
        pass

    @property
    @abstractmethod
    def height(self) -> float:
        """
        Height of extent in projected units.
        """
        pass

    @property
    @abstractmethod
    def cell_width(self) -> float:
        """
        Positive cell width in units of CRS.
        """
        pass

    @property
    @abstractmethod
    def cell_height(self) -> float:
        """
        Negative cell height in units of CRS.
        """
        pass

    @property
    @abstractmethod
    def x_min(self) -> float:
        pass

    @property
    @abstractmethod
    def x_max(self) -> float:
        pass

    @property
    @abstractmethod
    def y_min(self) -> float:
        pass

    @property
    @abstractmethod
    def y_max(self) -> float:
        pass

    @property
    def x_center(self) -> float:
        x_cent = (self.x_min + self.x_max) / 2.0

        if self.is_geographic and (x_cent < -180 or x_cent > 180):
            raise ValueError(f"invalid center longitude: {x_cent}")

        return x_cent

    @property
    def y_center(self) -> float:
        y_cent = (self.y_min + self.y_max) / 2.0

        if self.is_geographic and (y_cent < -90 or y_cent > 90):
            raise ValueError(f"invalid center latitude {y_cent}")

        return y_cent

    @property
    def centroid(self) -> Point:
        from .point import Point

        return Point(self.x_center, self.y_center, crs=self.crs)

    @property
    def centroid_latlon(self) -> Point:
        return self.centroid.latlon

    @property
    def center_aeqd(self) -> CRS:
        from .center_aeqd import center_aeqd

        return center_aeqd(self.centroid_latlon)

    def get_bbox(self, crs: Union[CRS, str] = None) -> BBox:
        from .CRS import CRS
        from .bbox import BBox

        if crs is None:
            crs = self.crs

        if not isinstance(crs, CRS):
            crs = CRS(crs)

        if crs == self.crs:
            boundary = self.boundary
        else:
            boundary = self.boundary.to_crs(crs)

        x, y = boundary.exterior.xy
        x = np.array(x)
        y = np.array(y)

        if crs.is_geographic and self.crosses_antimeridian:
            x_min = float(np.nanmin(x[x > 0]))
            y_min = float(np.nanmin(y))
            x_max = float(np.nanmax(x[x < 0]))
            y_max = float(np.nanmax(y))
        else:
            x_min = float(np.nanmin(x))
            y_min = float(np.nanmin(y))
            x_max = float(np.nanmax(x))
            y_max = float(np.nanmax(y))

        bbox = BBox(x_min, y_min, x_max, y_max, crs=crs)

        return bbox

    bbox = property(get_bbox)

    @property
    @abstractmethod
    def corner_polygon(self) -> Polygon:
        """
        Draw polygon through the corner coordinates of geolocation arrays.
        :return: shapely.geometry.Polygon of corner coordinate boundary
        """
        pass

    @property
    def corner_polygon_latlon(self) -> Polygon:
        return self.corner_polygon.to_crs(WGS84)

    def get_boundary(self, crs: Union[CRS, str] = None) -> Polygon:
        from .CRS import CRS

        if crs is None:
            crs = self.crs

        if not isinstance(crs, CRS):
            crs = CRS(crs)

        boundary = self.boundary
        transformed_boundary = self.boundary.to_crs(crs)

        return transformed_boundary

    @property
    @abstractmethod
    def boundary(self):
        pass

    @property
    def boundary_latlon(self) -> Polygon:
        return self.get_boundary(WGS84)

    @property
    def crosses_antimeridian(self) -> bool:
        if self._crosses_antimeridian is not None:
            return self._crosses_antimeridian

        def to_polar(lon, lat):
            phi = np.pi / 180. * lon
            radius = np.pi / 180. * (90. - sign * lat)

            # nudge points at +/- 180 out of the way so they don't intersect the testing wedge
            phi = np.sign(phi) * np.where(np.abs(phi) > np.pi - 1.5 * epsilon, np.pi - 1.5 * epsilon, abs(phi))
            radius = np.where(radius < 1.5 * epsilon, 1.5 * epsilon, radius)

            x = radius * np.sin(phi)
            y = radius * np.cos(phi)

            if (isinstance(lon, list)):
                x = x.tolist()
                y = y.tolist()
            elif (isinstance(lon, tuple)):
                x = tuple(x)
                y = tuple(y)

            return (x, y)

        epsilon = 1e-14

        antimeridian_wedge = shapely.geometry.Polygon([
            (epsilon, -np.pi),
            (epsilon ** 2, -epsilon),
            (0, epsilon),
            (-epsilon ** 2, -epsilon),
            (-epsilon, -np.pi),
            (epsilon, -np.pi)
        ])

        feature_shape = self.boundary_latlon.geometry
        sign = 2. * (0.5 * (feature_shape.bounds[1] + feature_shape.bounds[3]) >= 0.) - 1.
        polar_shape = shapely_transform(to_polar, feature_shape)

        self._crosses_antimeridian = polar_shape.intersects(antimeridian_wedge)

        return self._crosses_antimeridian

    # TODO update for VectorGeometry
    def intersects(self, geometry: Union[BaseGeometry, RasterGeometry]) -> bool:
        if isinstance(geometry, RasterGeometry):
            geometry = geometry.get_boundary(crs=self.crs)

        if hasattr(geometry, "geometry"):
            geometry = geometry.geometry

        if not isinstance(geometry, BaseGeometry):
            raise ValueError("invalid geometry for intersection")

        result = self.boundary.geometry.intersects(geometry)

        return result

    @property
    def geolocation(self) -> RasterGeolocation:
        from .raster_geolocation import RasterGeolocation

        return RasterGeolocation(x=self.x, y=self.y, crs=self.crs)

    def to_crs(self, crs: Union[CRS, str] = WGS84) -> RasterGeolocation:
        from .CRS import CRS
        from .raster_geolocation import RasterGeolocation

        # validate destination CRS
        if not isinstance(crs, CRS):
            crs = CRS(crs)

        if self.crs == crs:
            return self

        if self.is_geographic:
            x, y = shapely_transform(self.crs, crs, self.y, self.x)
        else:
            x, y = shapely_transform(self.crs, crs, self.x, self.y)

        if crs.is_geographic:
            x = np.where(np.logical_or(x < -180, x > 180), np.nan, x)
            y = np.where(np.logical_or(y < -90, y > 90), np.nan, y)

        geolocation = RasterGeolocation(x=x, y=y, crs=crs)

        return geolocation

    def shifted_distances(self, rows, cols) -> np.ndarray:
        x = self.x
        y = self.y
        x_shifted = shift(x, (rows, cols), cval=np.nan)
        lat_shifted = shift(y, (rows, cols), cval=np.nan)
        differences = np.stack([x.flatten(), y.flatten()]) - np.stack([x_shifted.flatten(), lat_shifted.flatten()])
        differences = differences[:, ~np.any(np.isnan(differences), axis=0)]
        distances = np.linalg.norm(differences, axis=0)

        return distances

    @property
    def row_distances(self) -> np.ndarray:
        return self.shifted_distances(1, 0)

    @property
    def col_distances(self) -> np.ndarray:
        return self.shifted_distances(0, 1)

    @property
    def cell_size(self) -> float:
        return min([self.cell_width, abs(self.cell_height)])

    def cell_size_proj(self, crs: Union[CRS, str] = None) -> float:
        if crs is None:
            crs = self.center_aeqd

        projected_boundary = self.get_boundary(crs)
        x, y = projected_boundary.exterior.xy
        cell_width = (np.nanmax(x) - np.nanmin(x)) / self.cols
        cell_height = (np.nanmax(y) - np.nanmin(y)) / self.rows
        cell_size = np.nanmax([cell_width, cell_height])

        return cell_size

    @property
    def cell_size_meters(self) -> float:
        EQUATOR_CIRCUMFERENCE = 40075017.0
        if self._cell_size_meters is not None:
            return self._cell_size_meters

        if not self.is_geographic:
            return self.cell_size

        self._cell_size_meters = self.cell_size / 180.0 * EQUATOR_CIRCUMFERENCE

        return self._cell_size_meters

    def grid_to_size(
            self,
            cell_size_meters: float,
            target_crs: Union[CRS, str] = None,
            intermediate_crs: Union[CRS, str] = None,
            adjust_cell_size: bool = False):
        from .CRS import CRS
        from .raster_grid import RasterGrid

        if intermediate_crs is None:
            intermediate_crs = self.center_aeqd

        if target_crs is None:
            target_crs = self.crs

        # validate destination CRS
        if not isinstance(target_crs, CRS):
            target_crs = CRS(target_crs)

        if np.isinf(cell_size_meters) or not cell_size_meters > 0:
            raise ValueError(f"invalid cell size {cell_size_meters}")

        # knowing spatial sampling rate in meters
        # calculate the number of rows and columns required to space evenly
        # between edges of extent
        x_min_cent, y_min_cent, x_max_cent, y_max_cent = self.bbox.transform(intermediate_crs)

        width_meters = x_max_cent - x_min_cent
        height_meters = y_max_cent - y_min_cent
        rows = max(int(height_meters / cell_size_meters), 1)
        cols = max(int(width_meters / cell_size_meters), 1)

        if not (rows > 0 and cols > 0):
            raise ValueError("invalid shape ({}, {}) calculated from width: {} height: {} cell: {} bbox: {}".format(
                rows,
                cols,
                width_meters,
                height_meters,
                cell_size_meters,
                (x_min_cent, y_min_cent, x_max_cent, y_max_cent)
            ))

        # now that you know the ideal dimensions, apply this size to the target coordinate system
        x_min_dest, y_min_dest, x_max_dest, y_max_dest = self.bbox.transform(target_crs)
        width_dest = x_max_dest - x_min_dest
        height_dest = y_max_dest - y_min_dest

        if adjust_cell_size or target_crs.is_geographic:
            cell_width = width_dest / cols
            cell_height = height_dest / rows
        else:
            cell_width = cell_size_meters
            cell_height = cell_size_meters

        # generate grid object
        grid = RasterGrid(x_min_dest, y_max_dest, cell_width, -cell_height, rows, cols, crs=target_crs)

        return grid

    def UTM(self, cell_size_meters: float) -> RasterGrid:
        crs = self.local_UTM_proj4

        grid = self.grid_to_size(
            cell_size_meters,
            target_crs=crs,
            intermediate_crs=crs
        )

        return grid

    def geographic(self, cell_size_degrees: float, snap=True) -> RasterGrid:
        from .raster_grid import RasterGrid

        crs = WGS84
        lon_min, lat_min, lon_max, lat_max = self.bbox.latlon

        if np.isinf(cell_size_degrees) or not cell_size_degrees > 0:
            raise ValueError(f"invalid cell size {cell_size_degrees}")

        width = lon_max - lon_min
        height = lat_max - lat_min
        rows = max(int(height / cell_size_degrees), 1)
        cols = max(int(width / cell_size_degrees), 1)

        if snap:
            lat_origin = int(lat_max / cell_size_degrees) * cell_size_degrees
            lon_origin = int(lon_min / cell_size_degrees) * cell_size_degrees
        else:
            lat_origin = lat_max
            lon_origin = lon_min

        # generate grid object
        grid = RasterGrid(
            lon_origin,
            lat_origin,
            cell_size_degrees,
            -cell_size_degrees,
            rows,
            cols,
            crs=crs
        )

        return grid

    def grid_to_shape(
            self,
            rows: int,
            cols: int,
            dest_crs: Union[CRS, str] = None) -> RasterGrid:
        from .CRS import CRS
        from .raster_grid import RasterGrid

        if dest_crs is None:
            dest_crs = self.crs

        # validate destination CRS
        if not isinstance(dest_crs, CRS):
            dest_crs = CRS(dest_crs)

        # now that you know the ideal dimensions, apply this size to the target coordinate system
        x_min_dest, y_min_dest, x_max_dest, y_max_dest = self.bbox.transform(dest_crs)
        width_dest = x_max_dest - x_min_dest
        height_dest = y_max_dest - y_min_dest

        cell_width = width_dest / cols
        cell_height = height_dest / rows

        # generate grid object
        grid = RasterGrid(x_min_dest, y_max_dest, cell_width, -cell_height, rows, cols, crs=dest_crs)

        return grid

    def grid_to_crs(
            self,
            target_crs: Union[CRS, str] = None,
            target_cell_size: float = None) -> RasterGrid:
        from .CRS import CRS
        from .raster_grid import RasterGrid

        if target_crs is None:
            target_crs = self.crs

        # validate destination CRS
        if not isinstance(target_crs, CRS):
            target_crs = CRS(target_crs)

        x = self.x
        y = self.y

        if self.is_geographic:
            warnings.filterwarnings('ignore')

            invalid = np.logical_or(
                np.logical_or(x < -180, x > 180),
                np.logical_or(y < -90, y > 90)
            )

            x = np.where(invalid, np.nan, x)
            y = np.where(invalid, np.nan, y)

            warnings.resetwarnings()

        # transform source geolocation arrays to destination CRS

        source_x_trans, source_y_trans = Transformer.from_crs(self.crs, target_crs).transform(x, y)

        # source_x_trans, source_y_trans = transform(source_crs, target_crs, x, y)

        if not (np.isfinite(source_x_trans).any() or np.isfinite(source_y_trans).any()):
            raise ValueError(
                f"transformed x {source_x_trans} and transformed y {source_y_trans} from x {x} and y {y}")

        # calculate average cell size of source coordinate field in destination CRS
        if target_cell_size is None:
            target_cell_size = self.to_crs(crs=target_crs).cell_size

        if not np.isfinite(target_cell_size):
            raise ValueError(f"invalid cell size from x {self.x} and y {self.y}")

        # calculate the western boundary of the grid
        x_min = np.nanmin(source_x_trans[np.isfinite(source_x_trans)]) - target_cell_size / 2.0

        if not np.isfinite(x_min):
            raise ValueError("invalid x minimum")

        # calculate the eastern boundary of the grid
        x_max = np.nanmax(source_x_trans[np.isfinite(source_x_trans)]) + target_cell_size / 2.0

        if not np.isfinite(x_max):
            raise ValueError("invalid x maximum")

        # calculate the width of the grid in projected units
        width = x_max - x_min

        if not np.isfinite(width):
            raise ValueError(f"width {width} from x max {x_max} and x min {x_min}")

        try:
            # calculate the number of array columns needed to represent the projected width at its cell size
            cols = int(width / target_cell_size)
        except Exception as e:
            raise ValueError(f"can't compute cols from width {width} and cell size {target_cell_size}")

        # calculate the southern boundary of the grid
        y_min = np.nanmin(source_y_trans[np.isfinite(source_y_trans)]) - target_cell_size / 2.0

        if not np.isfinite(y_min):
            raise ValueError("invalid y minimum")

        # calculate the northern boundary of the grid
        y_max = np.nanmax(source_y_trans[np.isfinite(source_y_trans)]) + target_cell_size / 2.0

        if not np.isfinite(y_max):
            raise ValueError("invalid y maximum")

        # calculate the height of the grid in projected units
        height = y_max - y_min

        # calculate the number of array rows needed to represent the projected height at its cell size
        rows = int(height / target_cell_size)

        # generate grid object
        grid = RasterGrid(x_min, y_max, target_cell_size, -target_cell_size, rows, cols, crs=target_crs)

        return grid

    def generate_grid(
            self,
            dest_crs: Union[CRS, str] = None,
            cell_size: float = None,
            cell_size_meters: float = None,
            rows: int = None,
            cols: int = None):
        """
        This method generates a projected grid in the same extent and resolution as a given swath.
        :param dest_crs: coordinate reference system of the desired CRS as proj4 string or pyproj.CRS
        :return: CoordinateGrid object
        """
        if rows is not None and cols is not None:
            return self.grid_to_shape(rows, cols, dest_crs=dest_crs)
        if cell_size_meters is not None:
            return self.grid_to_size(cell_size_meters, target_crs=dest_crs)
        else:
            return self.grid_to_crs(target_crs=dest_crs, target_cell_size=cell_size)

    def reproject(self, crs: Union[CRS, str], cell_size: float):
        return self.generate_grid(dest_crs=crs, cell_size=cell_size)

    @classmethod
    def from_dict(cls, parameters):
        """
        Parse dictionary of CRS parameters into RasterGeometry object.
        :return: new RasterGeolocation or RasterGrid object encapsulating parameters read from dict
        """

        from .CRS import CRS
        from .raster_geolocation import RasterGeolocation
        from .raster_grid import RasterGrid

        # TODO this should conform to the CoverageJSON standard

        if 'type' not in parameters:
            raise ValueError("CRS parameters must specify 'type' as 'swath' or 'grid'")

        geometry_type = parameters['type']

        try:
            geometry_type = geometry_type.decode()
        except:
            pass

        if 'crs' not in parameters:
            raise ValueError("CRS parameters must specify 'crs' as proj4 string")

        proj4_string = parameters['crs']

        try:
            proj4_string = proj4_string.decode()
        except:
            pass

        try:
            crs = CRS(proj4_string)
        except Exception as e:
            cls.logger.exception(e)
            raise ValueError(f"problem with CRS string: {proj4_string}")

        if geometry_type == 'swath':
            # validate coordinate field

            if 'x' in parameters and 'y' in parameters:
                x = np.array(parameters['x'])
                y = np.array(parameters['y'])
            elif crs.is_geographic and 'latitude' in parameters and 'longitude' in parameters:
                x = np.array(parameters['longitude'])
                y = np.array(parameters['latitude'])
            elif crs.is_geographic and 'lat' in parameters and 'lon' in parameters:
                x = np.array(parameters['lon'])
                y = np.array(parameters['lat'])
            else:
                raise ValueError("coordinate field must specify 'x' and 'y' geolocation arrays")

            return RasterGeolocation(crs=crs, x=x, y=y)

        elif geometry_type == 'grid':
            # validate coordinate grid

            if 'rows' not in parameters or 'cols' not in parameters:
                raise ValueError("coordinate grid must specify 'rows' and 'cols'")

            rows = int(parameters['rows'])
            cols = int(parameters['cols'])

            if 'cell_width' not in parameters:
                raise ValueError("coordinate grid must specify 'cell_width'")

            cell_width = float(parameters['cell_width'])

            if not cell_width > 0:
                raise ValueError("parameter 'cell_width' must be positive width of cell in units of CRS")

            if 'cell_height' not in parameters:
                raise ValueError("coordinate grid must specify 'cell_height")

            cell_height = float(parameters['cell_height'])

            if not cell_height < 0:
                raise ValueError("parameter 'cell_height' must be negative height of cell in units of CRS")

            if 'x_origin' not in parameters:
                raise ValueError("coordinate grid must specify 'x_origin' x-coordinate of top-left corner of extent")

            x_origin = float(parameters['x_origin'])

            if 'y_origin' not in parameters:
                raise ValueError("coordinate grid must specify 'y_origin' y-coordinate of top-left corner of extent")

            y_origin = float(parameters['y_origin'])

            return RasterGrid(
                x_origin=x_origin,
                y_origin=y_origin,
                cell_width=cell_width,
                cell_height=cell_height,
                rows=rows,
                cols=cols,
                crs=crs
            )
        else:
            raise ValueError("geometry {} not recognized, must specify 'swath' or 'grid'".format(geometry_type))

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @property
    def to_json(self) -> str:
        # TODO implement CoverageJSON
        # https://www.w3.org/TR/covjson-overview/
        raise NotImplementedError()

    def mosaic_geometry(
            self,
            source_geometries,
            source_cell_size_meters=None,
            target_cell_size_meters=None,
            rows=None,
            cols=None,
            mosaic_to_target_cell_size=False):

        if source_cell_size_meters is None:
            if self.is_geographic:
                # noinspection PyTypeChecker
                source_cell_size_meters = float(np.mean([
                    geometry.cell_size_meters
                    for geometry
                    in source_geometries
                ]))

        if source_cell_size_meters is not None and target_cell_size_meters is not None:
            if source_cell_size_meters == target_cell_size_meters:
                return self

        if self.shape == (rows, cols):
            return self

        if source_cell_size_meters is not None and target_cell_size_meters is not None:
            if mosaic_to_target_cell_size:
                mosaic_cell_size_meters = target_cell_size_meters
            else:
                mosaic_cell_size_meters = source_cell_size_meters

            if target_cell_size_meters > source_cell_size_meters:
                mosaic_cell_size_meters = target_cell_size_meters

        else:
            mosaic_cell_size_meters = None

        if mosaic_cell_size_meters is not None and np.isinf(mosaic_cell_size_meters):
            mosaic_cell_size_meters = None

        # mosaic geometry needs a buffer around it to prevent gaps at edge of scene
        mosaic_geometry = self.generate_grid(
            cell_size_meters=mosaic_cell_size_meters,
            rows=rows,
            cols=cols
        ).buffer(3)

        return mosaic_geometry

    # @profile
    def kd_tree(self, geometry: RasterGeometry) -> KDTree:
        from .kdtree import KDTree

        return KDTree(self, geometry)

    @property
    def pixel_centroids(self) -> MultiPoint:
        from .multi_point import MultiPoint

        x, y = self.xy
        pixel_centroids = wrap_geometry(MultiPoint(np.stack([x.flatten(), y.flatten()], axis=1)), crs=self.crs)
        pixel_centroids._crs = self.crs

        return pixel_centroids

    @property
    def pixel_outlines(self) -> MultiPolygon:
        from .multi_polygon import MultiPolygon

        x, y = self.xy
        x = x.flatten()
        y = y.flatten()
        x_min = x - self.cell_width / 2
        x_max = x + self.cell_width / 2
        y_min = y + self.cell_height / 2
        y_max = y - self.cell_height / 2

        ul = np.stack([x_min, y_max], axis=1)
        ur = np.stack([x_max, y_max], axis=1)
        lr = np.stack([x_max, y_min], axis=1)
        ll = np.stack([x_min, y_min], axis=1)
        stack = np.stack([ul, ur, lr, ll], axis=1)

        pixel_outlines = MultiPolygon(
            gpd.GeoDataFrame({}, geometry=[shapely.multipolygons(stack)]).geometry[0],
            crs=self.crs
        )

        return pixel_outlines
