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
from .bbox import BBox
from .point import Point
from .polygon import Polygon

if TYPE_CHECKING:
    from .CRS import CRS
    from .spatial_geometry import SpatialGeometry
    from .raster import Raster

class RasterGrid(RasterGeometry):
    """
    This class encapsulates the georeferencing of gridded data using affine transforms.
    Gridded surfaces are assumed to be north-oriented. Row and column rotation are not supported.
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
        """
        Initialize a RasterGrid object.

        Args:
            x_origin (float): X-coordinate of the top-left corner of the grid.
            y_origin (float): Y-coordinate of the top-left corner of the grid.
            cell_width (float): Width of each cell in the grid.
            cell_height (float): Height of each cell in the grid (negative for north-oriented grids).
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            crs (Union[CRS, str], optional): Coordinate reference system. Defaults to WGS84.
            **kwargs: Additional keyword arguments.
        """
        super(RasterGrid, self).__init__(crs=crs, **kwargs)

        # Assemble affine transform for the grid
        self._affine = Affine(cell_width, 0, x_origin, 0, cell_height, y_origin)

        # Store grid dimensions
        self._rows = int(rows)
        self._cols = int(cols)

        # Create blank geolocation attributes
        self._x = None
        self._y = None

    def _subset_index(self, y_slice: slice, x_slice: slice) -> RasterGrid:
        """
        Create a subset of the grid based on the provided slices.

        Args:
            y_slice (slice): Slice for the rows.
            x_slice (slice): Slice for the columns.

        Returns:
            RasterGrid: A new RasterGrid object representing the subset.
        """
        y_start, y_end, y_step = y_slice.indices(self.rows)
        x_start, x_end, x_step = x_slice.indices(self.cols)

        rows = y_end - y_start
        cols = x_end - x_start

        # Calculate new origins based on the slices
        y_origin = self.y_origin + y_start * self.cell_height if y_start > 0 else self.y_origin
        x_origin = self.x_origin + x_start * self.cell_width if x_start > 0 else self.x_origin

        # Create a new affine transform for the subset
        affine = Affine(
            self.affine.a,
            self.affine.b,
            x_origin,
            self.affine.d,
            self.affine.e,
            y_origin
        )

        # Create and return the subset grid
        subset = RasterGrid.from_affine(affine, rows, cols, self.crs)
        return subset

    def __eq__(self, other: RasterGrid) -> bool:
        """
        Check equality between two RasterGrid objects.

        Args:
            other (RasterGrid): Another RasterGrid object to compare.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        return (
            isinstance(other, RasterGrid) and
            self.crs == other.crs and
            self.affine == other.affine and
            self.shape == other.shape
        )

    @classmethod
    def from_affine(cls, affine: Affine, rows: int, cols: int, crs: Union[CRS, str] = WGS84) -> RasterGrid:
        """
        Create a RasterGrid from an affine transform.

        Args:
            affine (Affine): Affine transform object.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            crs (Union[CRS, str], optional): Coordinate reference system. Defaults to WGS84.

        Returns:
            RasterGrid: A new RasterGrid object.
        """
        if not isinstance(affine, Affine):
            raise ValueError("affine is not an Affine object")

        return RasterGrid(affine.c, affine.f, affine.a, affine.e, rows, cols, crs)

    @classmethod
    def from_rasterio(cls, file: DatasetReader, crs: Union[CRS, str] = None, **kwargs) -> RasterGrid:
        """
        Create a RasterGrid from a rasterio DatasetReader object.

        Args:
            file (DatasetReader): Rasterio dataset reader object.
            crs (Union[CRS, str], optional): Coordinate reference system. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            RasterGrid: A new RasterGrid object.
        """
        # Handle compatibility with different rasterio versions
        affine = file.affine if hasattr(file, "affine") else file.transform

        # Use the CRS from the file if not provided
        if crs is None:
            crs = file.crs if file.crs is not None else WGS84

        return cls.from_affine(affine, file.height, file.width, crs)

    @classmethod
    def from_raster_file(cls, filename: str, **kwargs) -> RasterGrid:
        """
        Create a RasterGrid from a raster file.

        Args:
            filename (str): Path to the raster file.
            **kwargs: Additional keyword arguments.

        Returns:
            RasterGrid: A new RasterGrid object.
        """
        os.environ["CPL_ZIP_ENCODING"] = "UTF-8"

        with rasterio.open(filename, "r", **kwargs) as file:
            return cls.from_rasterio(file, **kwargs)

    @classmethod
    def open(cls, filename: str, **kwargs) -> RasterGrid:
        """
        Open a raster file and create a RasterGrid.

        Args:
            filename (str): Path to the raster file.
            **kwargs: Additional keyword arguments.

        Returns:
            RasterGrid: A new RasterGrid object.
        """
        return cls.from_raster_file(filename=filename, **kwargs)

    @classmethod
    def from_vectors(cls, x_vector: np.ndarray, y_vector: np.ndarray, crs: Union[CRS, str] = WGS84) -> RasterGrid:
        """
        Create a RasterGrid from x and y coordinate vectors.

        Args:
            x_vector (np.ndarray): Array of x-coordinates.
            y_vector (np.ndarray): Array of y-coordinates.
            crs (Union[CRS, str], optional): Coordinate reference system. Defaults to WGS84.

        Returns:
            RasterGrid: A new RasterGrid object.
        """
        cols = len(x_vector)
        rows = len(y_vector)

        # Calculate cell dimensions
        cell_width = np.nanmean(np.diff(x_vector))
        cell_height = np.nanmean(np.diff(y_vector))

        # Calculate origins
        x_origin = x_vector[0] - cell_width / 2.0
        y_origin = y_vector[0] + cell_height / 2.0

        return RasterGrid(
            x_origin=x_origin,
            y_origin=y_origin,
            cell_width=cell_width,
            cell_height=cell_height,
            rows=rows,
            cols=cols,
            crs=crs
        )

    @classmethod
    def from_bbox(
            cls,
            bbox: Union[BBox, Tuple[float]] = None,
            shape: Tuple[int, int] = None,
            cell_size: float = None,
            cell_width: float = None,
            cell_height: float = None,
            crs: Union[CRS, str] = None,
            xmin: float = None,
            ymin: float = None,
            xmax: float = None,
            ymax: float = None):
        """
        Create a RasterGrid from a bounding box or individual coordinates.

        Args:
            bbox (Union[BBox, Tuple[float]], optional): Bounding box object or tuple (xmin, ymin, xmax, ymax).
            shape (Tuple[int, int], optional): Shape of the grid as (rows, cols). Defaults to None.
            cell_size (float, optional): Uniform cell size. Defaults to None.
            cell_width (float, optional): Width of each cell. Defaults to None.
            cell_height (float, optional): Height of each cell. Defaults to None.
            crs (Union[CRS, str], optional): Coordinate reference system. Defaults to None.
            xmin (float, optional): Minimum x-coordinate. Defaults to None.
            ymin (float, optional): Minimum y-coordinate. Defaults to None.
            xmax (float, optional): Maximum x-coordinate. Defaults to None.
            ymax (float, optional): Maximum y-coordinate. Defaults to None.

        Returns:
            RasterGrid: A new RasterGrid object.

        Raises:
            ValueError: If both bbox and individual coordinates are provided, or if required parameters are missing.
        """
        # Handle either bbox or individual coordinates
        if bbox is not None and any(coord is not None for coord in [xmin, ymin, xmax, ymax]):
            raise ValueError("Provide either bbox parameter or individual xmin/ymin/xmax/ymax, not both")
        
        if bbox is None and any(coord is None for coord in [xmin, ymin, xmax, ymax]):
            raise ValueError("When not providing bbox, all of xmin, ymin, xmax, ymax must be provided")
        
        if bbox is None and all(coord is None for coord in [xmin, ymin, xmax, ymax]):
            raise ValueError("Must provide either bbox parameter or xmin, ymin, xmax, ymax")

        if (cell_width is None or cell_height is None) and cell_size is not None:
            cell_width = cell_size
            cell_height = -cell_size

        if bbox is not None:
            if crs is None and isinstance(bbox, BBox):
                crs = bbox.crs
            xmin, ymin, xmax, ymax = bbox

        if crs is None:
            crs = WGS84

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
        """
        Merge multiple RasterGeometry objects into a single RasterGrid.

        Args:
            geometries (List[RasterGeometry]): List of RasterGeometry objects to merge.
            crs (CRS, optional): Coordinate reference system for the merged grid. Defaults to None.
            cell_size (float, optional): Cell size for the merged grid. Defaults to None.

        Returns:
            RasterGrid: A new RasterGrid object representing the merged geometries.
        """
        if crs is None:
            crs = geometries[0].crs

        geometries = [geometry.to_crs(crs) for geometry in geometries]
        bbox = BBox.merge([geometry.bbox for geometry in geometries], crs=crs)

        if cell_size is None:
            cell_size = min([geometry.cell_size for geometry in geometries])

        geometry = RasterGrid.from_bbox(bbox=bbox, cell_size=cell_size, crs=crs)

        return geometry

    def get_bbox(self, CRS: Union[CRS, str] = None) -> BBox:
        """
        Get the bounding box of the grid.

        Args:
            CRS (Union[CRS, str], optional): Target coordinate reference system. Defaults to None.

        Returns:
            BBox: Bounding box of the grid.
        """
        bbox = BBox(xmin=self.x_min, ymin=self.y_min, xmax=self.x_max, ymax=self.y_max, crs=self.crs)

        if CRS is not None:
            bbox = bbox.transform(CRS)

        return bbox

    bbox = property(get_bbox)

    @property
    def affine(self) -> Affine:
        """
        Get the affine transform of the top-left corners of cells.

        Returns:
            Affine: Affine transform object.
        """
        return self._affine

    @property
    def affine_center(self) -> Affine:
        """
        Get the affine transform of cell centroids.

        Returns:
            Affine: Affine transform object.
        """
        return self.affine @ Affine.translation(0.5, 0.5)

    @property
    def cell_width(self) -> float:
        """
        Get the positive cell width in units of the CRS.

        Returns:
            float: Cell width.
        """
        return self.affine.a

    @property
    def width(self) -> float:
        """
        Get the total width of the grid.

        Returns:
            float: Total width of the grid.
        """
        return self.cell_width * self.cols

    @property
    def x_origin(self) -> float:
        """
        Get the x-coordinate of the top-left corner of the grid extent.

        Returns:
            float: X-coordinate of the top-left corner.
        """
        return self.affine.c

    @property
    def cell_height(self) -> float:
        """
        Get the negative cell height in units of the CRS.

        Returns:
            float: Cell height.
        """
        return self.affine.e

    @property
    def height(self) -> float:
        """
        Get the total height of the grid.

        Returns:
            float: Total height of the grid.
        """
        return abs(self.cell_height) * self.rows

    @property
    def y_origin(self) -> float:
        """
        Get the y-coordinate of the top-left corner of the grid extent.

        Returns:
            float: Y-coordinate of the top-left corner.
        """
        return self.affine.f

    @property
    def rows(self) -> int:
        """
        Get the number of rows in the grid.

        Returns:
            int: Number of rows.
        """
        return self._rows

    @property
    def cols(self) -> int:
        """
        Get the number of columns in the grid.

        Returns:
            int: Number of columns.
        """
        return self._cols

    @property
    def xmin(self) -> float:
        """
        Get the minimum x-coordinate of the grid extent.

        Returns:
            float: Minimum x-coordinate.
        """
        return self.x_origin
    
    @property
    def xmax(self) -> float:
        """
        Get the maximum x-coordinate of the grid extent.

        Returns:
            float: Maximum x-coordinate.
        """
        return self.x_origin + self.width
    
    @property
    def ymin(self) -> float:
        """
        Get the minimum y-coordinate of the grid extent.

        Returns:
            float: Minimum y-coordinate.
        """
        return self.y_origin - self.height

    @property
    def ymax(self) -> float:
        """
        Get the maximum y-coordinate of the grid extent.

        Returns:
            float: Maximum y-coordinate.
        """
        return self.y_origin

    @property
    def grid(self) -> RasterGrid:
        """
        Get a copy of the current grid.

        Returns:
            RasterGrid: Copy of the current grid.
        """
        return RasterGrid.from_affine(self.affine, self.rows, self.cols, self.crs)

    @property
    def corner_polygon(self) -> Polygon:
        """
        Get a polygon representing the corners of the grid.

        Returns:
            Polygon: Polygon of the grid corners.
        """
        return Polygon([
            (self.x_origin, self.y_origin),
            (self.x_origin + self.width, self.y_origin),
            (self.x_origin + self.width, self.y_origin - self.height),
            (self.x_origin, self.y_origin - self.height)
        ], crs=self.crs)

    @property
    def corner_polygon_latlon(self) -> Polygon:
        """
        Get a polygon representing the corners of the grid in latitude/longitude.

        Returns:
            Polygon: Polygon of the grid corners in latitude/longitude.
        """
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
        """
        Get the boundary of the grid as a polygon.

        Returns:
            Polygon: Boundary polygon of the grid.
        """
        if self.shape == (1, 1):
            return Polygon(LinearRing([
                (self.x_min, self.y_max),
                (self.x_max, self.y_max),
                (self.x_max, self.y_min),
                (self.x_min, self.y_min)
            ]))

        y_indices, x_indices = self.boundary_indices
        x_boundary, y_boundary = self.affine_center @ (x_indices, y_indices)
        points = np.c_[x_boundary, y_boundary]
        boundary = Polygon(points, crs=self.crs)

        return boundary

    def resolution(self, cell_size: Union[float, Tuple[float, float]]) -> RasterGrid:
        """
        Adjust the resolution of the grid.

        Args:
            cell_size (Union[float, Tuple[float, float]]): New cell size as a single value or tuple (width, height).

        Returns:
            RasterGrid: A new RasterGrid with adjusted resolution.
        """
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
        """
        Resize the grid to new dimensions.

        Args:
            dimensions (Tuple[int, int]): New dimensions as (rows, cols).
            keep_square (bool, optional): Whether to maintain square cells. Defaults to True.

        Returns:
            RasterGeometry: A new RasterGeometry object with resized dimensions.
        """
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
        """
        Rescale the grid based on cell size or dimensions.

        Args:
            cell_size (float, optional): New cell size. Defaults to None.
            rows (int, optional): New number of rows. Defaults to None.
            cols (int, optional): New number of columns. Defaults to None.

        Returns:
            RasterGrid: A new RasterGrid with rescaled dimensions.
        """
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
        """
        Add a buffer around the grid.

        Args:
            pixels (int): Number of pixels to buffer.

        Returns:
            RasterGrid: A new RasterGrid with the buffer applied.
        """
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
        Get the vector of x-coordinates.

        Returns:
            np.ndarray: Array of x-coordinates.
        """
        return (self.affine_center @ (np.arange(self.cols), np.full(self.cols, 0, dtype=np.float32)))[0]

    @property
    def y_vector(self) -> np.ndarray:
        """
        Get the vector of y-coordinates.

        Returns:
            np.ndarray: Array of y-coordinates.
        """
        return (self.affine_center @ (np.full(self.rows, 0, dtype=np.float32), np.arange(self.rows)))[1]

    @property
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the geolocation arrays for x and y coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates.
        """
        return self.affine_center @ np.meshgrid(np.arange(self.cols), np.arange(self.rows))

    def index_point(self, point: Point) -> Tuple[int, int]:
        """
        Get the grid index of a point.

        Args:
            point (Point): Point to index.

        Returns:
            Tuple[int, int]: Grid index as (row, col).
        """
        native_point = point.to_crs(self.crs)
        x = native_point.x
        y = native_point.y
        col, row = ~self.affine_center @ (x, y)
        col = int(round(col))
        row = int(round(row))
        index = (row, col)

        return index

    def index(self, geometry: Union[SpatialGeometry, Tuple[float, float, float, float]]):
        """
        Get the grid indices for a geometry.

        Args:
            geometry (Union[SpatialGeometry, Tuple[float, float, float, float]]): Geometry to index.

        Returns:
            Tuple[slice, slice]: Slices representing the grid indices.
        """
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
        """
        Get the rasterio window for a geometry.

        Args:
            geometry (Union[SpatialGeometry, Tuple[float, float, float, float]]): Geometry to create a window for.
            buffer (int, optional): Buffer to apply to the window. Defaults to None.

        Returns:
            Window: Rasterio window object.
        """
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
        """
        Subset the grid based on a target geometry.

        Args:
            target (Union[Window, Point, Polygon, BBox, RasterGeometry]): Target geometry for subsetting.

        Returns:
            RasterGrid: Subset of the grid.
        """
        if not isinstance(target, Window):
            target = self.window(target)

        slices = target.toslices()
        subset = self[slices]

        return subset

    def shift_xy(self, x_shift: float, y_shift: float) -> RasterGrid:
        """
        Shift the grid by x and y distances.

        Args:
            x_shift (float): Distance to shift in the x-direction.
            y_shift (float): Distance to shift in the y-direction.

        Returns:
            RasterGrid: A new RasterGrid with the shift applied.
        """
        new_affine = self.affine @ Affine.translation(x_shift / self.cell_size, y_shift / self.cell_size)
        grid = RasterGrid.from_affine(new_affine, self.rows, self.cols, self.crs)

        return grid

    def shift_distance(self, distance: float, direction: float) -> RasterGrid:
        """
        Shift the grid by a distance in a specific direction.

        Args:
            distance (float): Distance to shift.
            direction (float): Direction to shift in degrees.

        Returns:
            RasterGrid: A new RasterGrid with the shift applied.
        """
        x_shift = distance * np.cos(np.radians(direction))
        y_shift = distance * np.sin(np.radians(direction))
        grid = self.shift_xy(x_shift, y_shift)

        return grid

    @property
    def x(self) -> np.ndarray:
        """
        Get the geolocation array of x-coordinates.

        Returns:
            np.ndarray: Array of x-coordinates.
        """
        # cache x-coordinate array
        if self._x is None:
            self._x = self.xy[0]

        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        Get the geolocation array of y-coordinates.

        Returns:
            np.ndarray: Array of y-coordinates.
        """
        # cache y-coordinate array
        if self._y is None:
            self._y = self.xy[1]

        return self._y

    @property
    def x_min(self) -> float:
        """
        Get the western boundary of the grid extent.

        Returns:
            float: Western boundary.
        """
        return self.x_origin

    @property
    def x_max(self) -> float:
        """
        Get the eastern boundary of the grid extent.

        Returns:
            float: Eastern boundary.
        """
        return self.x_origin + self.width

    @property
    def y_min(self) -> float:
        """
        Get the southern boundary of the grid extent.

        Returns:
            float: Southern boundary.
        """
        return self.y_origin - self.height

    @property
    def y_max(self) -> float:
        """
        Get the northern boundary of the grid extent.

        Returns:
            float: Northern boundary.
        """
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
        """
        Rasterize shapes into the grid.

        Args:
            shapes (Iterable): Shapes to rasterize.
            shape_crs (optional): CRS of the shapes. Defaults to None.
            fill (int, optional): Fill value for empty cells. Defaults to 0.
            all_touched (bool, optional): Whether to rasterize all touched cells. Defaults to False.
            merge_alg (MergeAlg, optional): Merge algorithm. Defaults to MergeAlg.replace.
            default_value (int, optional): Default value for shapes. Defaults to 1.
            dtype (optional): Data type of the raster. Defaults to None.

        Returns:
            Raster: Rasterized representation of the shapes.
        """
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
        """
        Mask the grid using geometries.

        Args:
            geometries (gpd.GeoDataFrame): Geometries to mask with.
            all_touched (bool, optional): Whether to mask all touched cells. Defaults to False.
            invert (bool, optional): Whether to invert the mask. Defaults to False.

        Returns:
            RasterGrid: Masked grid.
        """
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
        """
        Get the coverage metadata of the grid.

        Returns:
            OrderedDict: Coverage metadata.
        """
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

    def to_dict(self, output_dict: dict = None, write_geolocation_arrays: bool = False) -> dict:
        """
        Convert the RasterGrid to a dictionary representation.

        Args:
            output_dict (dict, optional): Dictionary to populate. Defaults to None.
            write_geolocation_arrays (bool, optional): Whether to include geolocation arrays. Defaults to False.

        Returns:
            dict: Dictionary representation of the RasterGrid.
        """
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
