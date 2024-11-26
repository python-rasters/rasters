from __future__ import annotations

from typing import Union, Tuple, List, Iterable, TYPE_CHECKING

import os

from collections import OrderedDict

import numpy as np

from shapely.geometry import LinearRing

import geopandas as gpd

from affine import Affine

import rasterio
from rasterio import DatasetReader
from rasterio.windows import Window
from rasterio.enums import MergeAlg

from .out_of_bounds_error import OutOfBoundsError

from .CRS import WGS84

from .raster_geometry import RasterGeometry
from .wrap_geometry import wrap_geometry

if TYPE_CHECKING:
    from .CRS import CRS
    from .spatial_geometry import SpatialGeometry
    from .bbox import BBox
    from .point import Point
    from .polygon import Polygon
    from .raster import Raster

class RasterGrid(RasterGeometry):
    """
    This class encapsulates the georeferencing of gridded data using affine transforms.
    Gridded surfaces are assumed north-oriented. Row and column rotation are not supported.
    """
    geometry_type = "grid"

    def __init__(
            self,
            x_origin: float,
            y_origin: float,
            cell_width: float,
            cell_height: float,
            rows: int,
            cols: int,
            crs: Union[CRS, str] = WGS84,
            **kwargs):
        super(RasterGrid, self).__init__(crs=crs, **kwargs)

        # assemble affine transform
        self._affine = Affine(cell_width, 0, x_origin, 0, cell_height, y_origin)

        # store dimensions
        self._rows = int(rows)
        self._cols = int(cols)

        # create blank geolocation attributes
        self._x = None
        self._y = None

    def _subset_index(self, y_slice: slice, x_slice: slice) -> RasterGrid:
        y_start, y_end, y_step = y_slice.indices(self.rows)
        x_start, x_end, x_step = x_slice.indices(self.cols)

        rows = y_end - y_start
        cols = x_end - x_start

        if y_start > 0:
            y_origin = self.y_origin + y_start * self.cell_height
        else:
            y_origin = self.y_origin

        if x_start > 0:
            x_origin = self.x_origin + x_start * self.cell_width
        else:
            x_origin = self.x_origin

        affine = Affine(
            self.affine.a,
            self.affine.b,
            x_origin,
            self.affine.d,
            self.affine.e,
            y_origin
        )

        subset = RasterGrid.from_affine(affine, rows, cols, self.crs)

        return subset

    def __eq__(self, other: RasterGrid) -> bool:
        return isinstance(other,
                          RasterGrid) and self.crs == other.crs and self.affine == other.affine and self.shape == other.shape

    @classmethod
    def from_affine(cls, affine: Affine, rows: int, cols: int, crs: Union[CRS, str] = WGS84):
        if not isinstance(affine, Affine):
            raise ValueError("affine is not an Affine object")

        return RasterGrid(affine.c, affine.f, affine.a, affine.e, rows, cols, crs)

    @classmethod
    def from_rasterio(cls, file: DatasetReader, crs: Union[CRS, str] = None, **kwargs) -> RasterGrid:
        # FIXME old version of rasterio uses affine, new version uses transform

        if hasattr(file, "affine"):
            affine = file.affine
        else:
            affine = file.transform

        if crs is None:
            if file.crs is None:
                crs = WGS84
            else:
                crs = file.crs

        return cls.from_affine(affine, file.height, file.width, crs)

    @classmethod
    def from_raster_file(cls, filename: str, **kwargs) -> RasterGrid:
        os.environ["CPL_ZIP_ENCODING"] = "UTF-8"

        with rasterio.open(filename, "r", **kwargs) as file:
            return cls.from_rasterio(file, **kwargs)

    @classmethod
    def open(cls, filename: str, **kwargs) -> RasterGrid:
        return cls.from_raster_file(filename=filename, **kwargs)

    @classmethod
    def from_vectors(cls, x_vector: np.ndarray, y_vector: np.ndarray, crs: Union[CRS, str] = WGS84):
        cols = len(x_vector)
        rows = len(y_vector)

        cell_width = np.nanmean(np.diff(x_vector))
        cell_height = np.nanmean(np.diff(y_vector))

        x_origin = x_vector[0] - cell_width / 2.0
        y_origin = y_vector[0] + cell_height / 2.0

        grid = RasterGrid(
            x_origin=x_origin,
            y_origin=y_origin,
            cell_width=cell_width,
            cell_height=cell_height,
            rows=rows,
            cols=cols,
            crs=crs
        )

        return grid

    @classmethod
    def from_bbox(
            cls,
            bbox: Union[BBox, Tuple[float]],
            shape: Tuple[int, int] = None,
            cell_size: float = None,
            cell_width: float = None,
            cell_height: float = None,
            crs: Union[CRS, str] = None):
        from .bbox import BBox

        if (cell_width is None or cell_height is None) and cell_size is not None:
            cell_width = cell_size
            cell_height = -cell_size

        if crs is None and isinstance(bbox, BBox):
            crs = bbox.crs

        if crs is None:
            crs = WGS84

        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        x_origin = xmin
        y_origin = ymax

        if shape is None:
            if cell_width is None or cell_height is None:
                raise ValueError("no cell size given")

            cell_width = float(cell_width)
            cell_height = float(cell_height)
            cols = width / cell_width
            rows = height / abs(cell_height)
        else:
            rows, cols = shape
            cell_width = width / cols
            cell_height = -height / rows

        grid = RasterGrid(
            x_origin=x_origin,
            y_origin=y_origin,
            cell_width=cell_width,
            cell_height=cell_height,
            rows=rows,
            cols=cols,
            crs=crs
        )

        return grid

    @classmethod
    def merge(cls, geometries: List[RasterGeometry], crs: CRS = None, cell_size: float = None) -> RasterGrid:
        from .bbox import BBox
        
        if crs is None:
            crs = geometries[0].crs

        geometries = [geometry.to_crs(crs) for geometry in geometries]
        bbox = BBox.merge([geometry.bbox for geometry in geometries], crs=crs)

        if cell_size is None:
            cell_size = min([geometry.cell_size for geometry in geometries])

        geometry = RasterGrid.from_bbox(bbox=bbox, cell_size=cell_size, crs=crs)

        return geometry

    def get_bbox(self, CRS: Union[CRS, str] = None) -> BBox:
        from .bbox import BBox

        bbox = BBox(xmin=self.x_min, ymin=self.y_min, xmax=self.x_max, ymax=self.y_max, crs=self.crs)

        if CRS is not None:
            bbox = bbox.transform(CRS)

        return bbox

    bbox = property(get_bbox)

    @property
    def affine(self) -> Affine:
        """
        Affine transform of top-left corners of cells.
        """
        return self._affine

    @property
    def affine_center(self) -> Affine:
        """
        Affine transform of cell centroids.
        """
        return self.affine * Affine.translation(0.5, 0.5)

    @property
    def cell_width(self) -> float:
        """
        Positive cell width in units of CRS.
        """
        return self.affine.a

    @property
    def width(self) -> float:
        return self.cell_width * self.cols

    @property
    def x_origin(self) -> float:
        """
        X-coordinate of top-left corner of extent.
        """
        return self.affine.c

    @property
    def cell_height(self) -> float:
        """
        Negative cell height in units of CRS.
        """
        return self.affine.e

    @property
    def height(self) -> float:
        return abs(self.cell_height) * self.rows

    @property
    def y_origin(self) -> float:
        """
        Y-coordinate of top-left corner of extent.
        """
        return self.affine.f

    @property
    def rows(self) -> int:
        """
        Height of gridded surface in rows.
        """
        return self._rows

    @property
    def cols(self) -> int:
        """
        Width of gridded surface in columns.
        """
        return self._cols

    @property
    def grid(self) -> RasterGrid:
        return RasterGrid.from_affine(self.affine, self.rows, self.cols, self.crs)

    @property
    def corner_polygon(self) -> Polygon:
        """
        Draw polygon through the corner coordinates of geolocation arrays.
        :return: shapely.geometry.Polygon of corner coordinate boundary
        """
        from .polygon import Polygon

        return Polygon([
            (self.x_origin, self.y_origin),
            (self.x_origin + self.width, self.y_origin),
            (self.x_origin + self.width, self.y_origin - self.height),
            (self.x_origin, self.y_origin - self.height)
        ], crs=self.crs)

    @property
    def corner_polygon_latlon(self) -> Polygon:
        """
        Draw polygon through the corner coordinates of geolocation arrays in lat/lon.
        :return: shapely.geometry.Polygon of corner coordinate boundary
        """
        from .polygon import Polygon

        polygon = Polygon([
            (self.x_origin, self.y_origin),
            (self.x_origin + self.width, self.y_origin),
            (self.x_origin + self.width, self.y_origin - self.height),
            (self.x_origin, self.y_origin - self.height)
        ], crs=self.crs)

        polygon_latlon = polygon.to_crs(WGS84)

        return polygon_latlon

    @property
    def boundary(self) -> Polygon:
        from .polygon import Polygon

        if self.shape == (1, 1):
            return Polygon(LinearRing([
                (self.x_min, self.y_max),
                (self.x_max, self.y_max),
                (self.x_max, self.y_min),
                (self.x_min, self.y_min)
            ]))

        y_indices, x_indices = self.boundary_indices
        x_boundary, y_boundary = self.affine_center * (x_indices, y_indices)
        points = np.c_[x_boundary, y_boundary]
        boundary = Polygon(points, crs=self.crs)

        return boundary

    def resolution(self, cell_size: Union[float, Tuple[float, float]]) -> RasterGrid:
        if len(cell_size) == 1:
            cell_width = cell_size
            cell_height = -cell_size
        elif len(cell_size) == 2:
            cell_width, cell_height = cell_size
        else:
            raise ValueError(f"invalid cell size: {cell_size}")

        rows = int(self.height / abs(cell_height))
        cols = int(self.width / cell_width)
        affine = self.affine
        new_affine = Affine(cell_width, affine.b, affine.c, affine.d, cell_height, affine.f)
        grid = RasterGrid.from_affine(new_affine, rows, cols, crs=self.crs)

        return grid

    def resize(self, dimensions: Tuple[int, int], keep_square: bool = True) -> RasterGeometry:
        rows, cols = dimensions
        cell_height = self.cell_height * (float(self.rows) / float(rows))
        cell_width = self.cell_width * (float(self.cols) / float(cols))

        if abs(cell_height) != cell_width:
            cell_height = -cell_width

        resized_grid = RasterGrid(
            self.x_origin,
            self.y_origin,
            cell_width,
            cell_height,
            rows,
            cols,
            self.crs
        )

        return resized_grid

    def rescale(self, cell_size: float = None, rows: int = None, cols: int = None):
        if rows is None and cols is None:
            rows = int(self.height / cell_size)
            cols = int(self.width / cell_size)

        if cell_size is None:
            cell_width = self.width / cols
            cell_height = -1 * (self.height / rows)
        else:
            cell_width = cell_size
            cell_height = -cell_size

        grid = RasterGrid(
            x_origin=self.x_origin,
            y_origin=self.y_origin,
            cell_width=cell_width,
            cell_height=cell_height,
            rows=rows,
            cols=cols,
            crs=self.crs
        )

        return grid

    def buffer(self, pixels: int) -> RasterGrid:
        return RasterGrid(
            x_origin=self.x_origin - (pixels * self.cell_width),
            y_origin=self.y_origin - (pixels * self.cell_height),
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            rows=self.rows + pixels * 2,
            cols=self.cols + pixels * 2,
            crs=self.crs
        )

    @property
    def x_vector(self) -> np.ndarray:
        """
        Vector of x-coordinates.
        """
        return (self.affine_center * (np.arange(self.cols), np.full(self.cols, 0, dtype=np.float32)))[0]

    @property
    def y_vector(self) -> np.ndarray:
        """
        Vector of y-coordinates.
        """
        return (self.affine_center * (np.full(self.rows, 0, dtype=np.float32), np.arange(self.rows)))[1]

    @property
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Geolocation arrays.
        """
        return self.affine_center * np.meshgrid(np.arange(self.cols), np.arange(self.rows))

    def index_point(self, point: Point) -> Tuple[int, int]:
        native_point = point.to_crs(self.crs)
        x = native_point.x
        y = native_point.y
        col, row = ~self.affine_center * (x, y)
        col = int(round(col))
        row = int(round(row))
        index = (row, col)

        return index

    def index(self, geometry: Union[SpatialGeometry, Tuple[float, float, float, float]]):
        from .point import Point

        geometry = wrap_geometry(geometry)
        xmin, ymin, xmax, ymax = geometry.bbox.transform(self.crs)

        row_start, col_start = self.index_point(Point(xmin, ymax, crs=self.crs))
        row_end, col_end = self.index_point(Point(xmax, ymin, crs=self.crs))
        row_end += 1
        col_end += 1

        rows, cols = self.shape

        if row_end < 0 or col_end < 0 or row_start > rows or col_start > cols:
            raise OutOfBoundsError(
                f"target geometry is not within source geometry row_start: {row_start} row_end: {row_end} col_start: {col_start} col_end: {col_end} rows: {rows} cols: {cols}\nsource geometry:\n{self}\ntarget geometry:\n{geometry}")

        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_end = min(row_end, rows)
        col_end = min(col_end, cols)

        index = (slice(row_start, row_end), slice(col_start, col_end))

        return index

    def window(
            self,
            geometry: Union[SpatialGeometry, (float, float, float, float)],
            buffer: int = None) -> Window:
        from .point import Point

        geometry = wrap_geometry(geometry)
        xmin, ymin, xmax, ymax = geometry.bbox.transform(self.crs)

        row_start, col_start = self.index_point(Point(xmin, ymax, crs=self.crs))
        row_end, col_end = self.index_point(Point(xmax, ymin, crs=self.crs))
        row_end += 1
        col_end += 1

        rows, cols = self.shape

        if row_end < 0 or col_end < 0 or row_start > rows or col_start > cols:
            raise OutOfBoundsError("target geometry is not within source geometry")

        if buffer is not None:
            row_start -= buffer
            col_start -= buffer
            row_end += buffer
            col_end += buffer

        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_end = min(row_end, rows)
        col_end = min(col_end, cols)

        window = Window(
            col_off=col_start,
            row_off=row_start,
            width=(col_end - col_start),
            height=(row_end - row_start)
        )

        return window

    def subset(self, target: Union[Window, Point, Polygon, BBox, RasterGeometry]) -> RasterGrid:
        if not isinstance(target, Window):
            target = self.window(target)

        slices = target.toslices()
        subset = self[slices]

        return subset
    
    def shift_xy(self, x_shift: float, y_shift: float) -> RasterGrid:
        new_affine = self.affine * Affine.translation(x_shift / self.cell_size, y_shift / self.cell_size)
        grid = RasterGrid.from_affine(new_affine, self.rows, self.cols, self.crs)

        return grid
    
    def shift_distance(self, distance: float, direction: float) -> RasterGrid:
        x_shift = distance * np.cos(np.radians(direction))
        y_shift = distance * np.sin(np.radians(direction))
        grid = self.shift_xy(x_shift, y_shift)

        return grid

    @property
    def x(self) -> np.ndarray:
        """
        Geolocation array of x-coordinates.
        """
        # cache x-coordinate array
        if self._x is None:
            self._x = self.xy[0]

        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        Geolocation array of y-coordinates.
        """
        # cache y-coordinate array
        if self._y is None:
            self._y = self.xy[1]

        return self._y

    @property
    def x_min(self) -> float:
        """
        Western boundary of extent.
        """
        return self.x_origin

    @property
    def x_max(self) -> float:
        """
        Eastern boundary of extent.
        """
        return self.x_origin + self.width

    @property
    def y_min(self) -> float:
        return self.y_origin - self.height

    @property
    def y_max(self) -> float:
        return self.y_origin

    def rasterize(
            self,
            shapes,
            shape_crs=None,
            fill=0,
            all_touched=False,
            merge_alg=MergeAlg.replace,
            default_value=1,
            dtype=None) -> Raster:
        from .raster import Raster

        if not isinstance(shapes, Iterable):
            shapes = [shapes]

        shapes = [
            wrap_geometry(shape, crs=shape_crs).to_crs(self.crs)
            for shape
            in shapes
        ]

        # TODO check in on the `rasterio.features` reference
        image = Raster(
            rasterio.features.rasterize(
                shapes=shapes,
                out_shape=self.shape,
                fill=fill,
                transform=self.affine,
                all_touched=all_touched,
                merge_alg=merge_alg,
                default_value=default_value,
                dtype=dtype
            ),
            geometry=self
        )

        return image

    def mask(
            self,
            geometries: gpd.GeoDataFrame,
            all_touched: bool = False,
            invert: bool = False) -> RasterGrid:
        from .raster import Raster

        mask_array = rasterio.features.geometry_mask(
            geometries,
            self.shape,
            self.affine,
            all_touched=all_touched,
            invert=invert
        )

        mask_raster = Raster(mask_array, geometry=self)

        return mask_raster

    @property
    def coverage(self) -> OrderedDict:
        coverage = OrderedDict()
        coverage["type"] = "Coverage"
        domain = OrderedDict()
        domain["type"] = "Domain"
        domain["domainType"] = "Grid"
        axes = OrderedDict()
        x = OrderedDict()
        x["start"] = self.x_min + self.cell_width / 2
        x["stop"] = self.x_max - self.cell_width / 2
        x["num"] = self.cols
        axes["x"] = x
        y = OrderedDict()
        y["start"] = self.y_min - self.cell_height / 2
        y["stop"] = self.y_max + self.cell_height / 2
        y["num"] = self.rows
        axes["y"] = y
        domain["axes"] = axes
        coverage["domain"] = domain
        coverage["referencing"] = [self.crs.coverage]

        return coverage

    def to_dict(self, output_dict=None, write_geolocation_arrays=False):
        if output_dict is None:
            output_dict = {}

        output_dict['type'] = 'grid'
        output_dict['crs'] = self.proj4
        output_dict['cell_width'] = self.cell_width
        output_dict['cell_height'] = self.cell_height
        output_dict['x_origin'] = self.x_origin
        output_dict['y_origin'] = self.y_origin
        output_dict['rows'] = self.rows
        output_dict['cols'] = self.cols

        if write_geolocation_arrays:
            lat, lon = self.latlon_matrices
            output_dict['latitude'] = lat
            output_dict['longitude'] = lon

        return output_dict
