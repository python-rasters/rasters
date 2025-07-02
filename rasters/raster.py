from __future__ import annotations

import io
import logging
import os
import warnings
from os import makedirs
from os.path import expanduser, exists, abspath, dirname, splitext
from typing import Any, Union, Tuple, List, TYPE_CHECKING

import PIL.Image
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
import skimage
from PIL.Image import Image
from astropy.visualization import MinMaxInterval, AsymmetricPercentileInterval
from matplotlib import colors
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.warp import reproject
from rasterio.windows import Window

from .CRS import WGS84
from .constants import *
from .out_of_bounds_error import OutOfBoundsError
from .raster_geolocation import RasterGeolocation
from .raster_geometry import RasterGeometry
from .raster_grid import RasterGrid
from .where import where
from .wrap_geometry import wrap_geometry
from .kdtree import KDTree

if TYPE_CHECKING:
    from .CRS import CRS
    from .bbox import BBox
    from .vector_geometry import VectorGeometry, SingleVectorGeometry
    from .point import Point
    from .multi_point import MultiPoint
    from .polygon import Polygon
    from .kdtree import KDTree
    from .multi_raster import MultiRaster

class Raster:
    DEFAULT_GEOTIFF_COMPRESSION = "deflate"

    multi = False

    logger = logging.getLogger(__name__)

    def __init__(
            self,
            array: Union[np.ndarray, Raster],
            geometry: RasterGeometry,
            nodata=None,
            cmap=None,
            metadata: dict = None,
            **kwargs):
        ALLOWED_ARRAY_TYPES = (
            np.ndarray,
            h5py.Dataset
        )

        if isinstance(array, Raster):
            array = array.array

        if not isinstance(array, ALLOWED_ARRAY_TYPES):
            raise TypeError('data is not a valid numpy.ndarray')

        ndim = len(array.shape)

        if ndim != 2:
            raise ValueError('data is not a two-dimensional array')

        dtype = array.dtype

        if dtype in (np.float32, np.float64):
            array = np.where(array == nodata, np.nan, array)

        self._array = array

        if nodata is None:
            if dtype in (np.float32, np.float64):
                nodata = np.nan

        self._nodata = nodata
        self._metadata = {}

        if metadata is not None:
            for key, value in metadata.items():
                self._metadata[key] = value

        for key, value in kwargs.items():
            if key == "metadata":
                continue

            self._metadata[key] = value

        if isinstance(geometry, RasterGeometry):
            self._geometry = geometry
        else:
            raise ValueError(f"geometry is not a valid RasterGeometry object: {type(geometry)}")

        self.cmap = cmap
        self._source_metadata = {}

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def dtype(self):
        return self._array.dtype

    def __array_prepare__(self, other: Union[np.ndarray, Raster], *args, **kwargs) -> np.ndarray:
        if isinstance(other, Raster):
            array = other.array
        elif isinstance(other, np.ndarray):
            array = other
        else:
            raise ValueError("cannot prepare array")

        return array

    def __array_wrap__(self, other: Union[np.ndarray, Raster], **kwargs) -> Raster:
        if isinstance(other, Raster):
            other = other.array
        elif isinstance(other, np.ndarray):
            other = other
        else:
            raise ValueError("cannot prepare array")

        return self.contain(other)

    def __array_finalize__(self, other: Union[np.ndarray, Raster], **kwargs) -> Raster:
        if isinstance(other, Raster):
            other = other.array
        elif isinstance(other, np.ndarray):
            other = other
        else:
            raise ValueError("cannot prepare array")

        return self.contain(other)

    def __add__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__add__(data, *args, **kwargs))

        return result

    __radd__ = __add__

    def __sub__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__sub__(data, *args, **kwargs))

        return result

    def __rsub__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        return self.contain(other - self.array)

    def __neg__(self):
        return self.contain(-(self.array))

    def __mul__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__mul__(data, *args, **kwargs))

        return result

    __rmul__ = __mul__

    def __pow__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__pow__(data, *args, **kwargs))

        return result

    def __rpow__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(data ** self.array)

        return result

    def __div__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__div__(data, *args, **kwargs))

        return result

    def __truediv__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__truediv__(data, *args, **kwargs))

        return result

    def __rtruediv__(self, other):
        return self.contain(other / self.array)

    def __floordiv__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__floordiv__(data, *args, **kwargs))

        return result

    def __mod__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__mod__(data, *args, **kwargs))

        return result

    def __lshift__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__lshift__(data, *args, **kwargs))

        return result

    def __rshift__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__rshift__(data, *args, **kwargs))

        return result

    def __and__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__and__(data, *args, **kwargs))

        return result

    def __or__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__or__(data, *args, **kwargs))

        return result

    def __xor__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__xor__(data, *args, **kwargs))

        return result

    def __invert__(self, *args, **kwargs) -> Raster:
        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__invert__(*args, **kwargs))

        return result

    def __lt__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__lt__(data, *args, **kwargs))

        return result

    def __le__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__le__(data, *args, **kwargs))

        return result

    def __gt__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__gt__(data, *args, **kwargs))

        return result

    def __ge__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__ge__(data, *args, **kwargs))

        return result

    def __cmp__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__cmp__(data, *args, **kwargs))

        return result

    def __eq__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__eq__(data, *args, **kwargs))

        return result

    def __ne__(self, other: Union[Raster, np.ndarray], *args, **kwargs) -> Raster:
        if isinstance(other, Raster):
            data = other.array
        else:
            data = other

        with np.errstate(invalid='ignore'):
            result = self.contain(self.array.__ne__(data, *args, **kwargs))

        return result

    def color(self, cmap) -> Raster:
        self.cmap = cmap

        return self

    def astype(self, type) -> Raster:
        return self.contain(self._array.astype(type))

    @classmethod
    def open(
            cls,
            filename: str,
            nodata=None,
            remove=None,
            geometry: Union[RasterGeometry, Point, MultiPoint] = None,
            buffer: int = None,
            window: Window = None,
            resampling: str = None,
            cmap: Union[Colormap, str] = None,
            **kwargs) -> Union[Raster, np.ndarray]:
        from .point import Point
        from .multi_point import MultiPoint
        from .raster_grid import RasterGrid

        if isinstance(geometry, MultiPoint):
            # First, we need to get the source raster's CRS to transform points if needed
            source_geometry = RasterGrid.open(filename)
            
            values = []

            for i, point in enumerate(geometry):
                # Transform point to match raster's CRS if they differ
                transformed_point = point
                if point.crs != source_geometry.crs:
                    transformed_point = point.to_crs(source_geometry.crs)
                
                try:
                    value = cls.open(
                        filename=filename,
                        nodata=nodata,
                        remove=remove,
                        geometry=transformed_point,
                        buffer=buffer,
                        window=window,
                        resampling=resampling,
                        cmap=cmap,
                        **kwargs
                    )

                    if isinstance(value, Raster):
                        # Use the original point (in original CRS) for the to_point conversion
                        value = value.to_point(point)

                    values.append(value)
                
                except OutOfBoundsError:
                    # Point is outside raster bounds, append NaN
                    values.append(np.nan)
                
                except Exception as e:
                    # For other exceptions, re-raise with more context about which point failed
                    raise type(e)(f"Point {i} at {point} (transformed to {transformed_point}) failed: {str(e)}") from e

            values = np.array(values)

            return values

        target_geometry = geometry

        if filename.startswith("~"):
            filename = expanduser(filename)

        if ":" not in filename and not exists(filename):
            raise IOError(f"raster file not found: {filename}")

        source_geometry = RasterGrid.open(filename)

        if buffer is None and isinstance(target_geometry, Point):
            buffer = 1

        if window is None and target_geometry is not None:
            window = source_geometry.window(geometry=target_geometry, buffer=buffer)

        if window is not None and not isinstance(window, Window):
            raise ValueError("invalid window")

        with rasterio.open(filename, "r", **kwargs) as file:
            if file.count != 1 and not cls.multi:
                raise IOError(f"raster file with {file.count} bands is not single-band: {filename}")

            if nodata is None:
                nodata = file.nodata

            if window is None:
                if cls.multi:
                    data = file.read()
                else:
                    data = file.read(1)
                geometry = source_geometry
            else:
                data = file.read(1, window=window)

                if remove is not None:
                    data = np.where(data == remove, nodata, data)

                rows, cols = data.shape
                CRS = file.crs
                target_affine = file.window_transform(window)

                geometry = RasterGrid.from_affine(
                    affine=target_affine,
                    rows=rows,
                    cols=cols,
                    crs=CRS
                )

        result = cls(data, geometry, nodata=nodata, filename=filename, cmap=cmap, **kwargs)

        if isinstance(target_geometry, RasterGeometry):
            result = result.to_geometry(target_geometry, resampling=resampling)
        elif isinstance(target_geometry, Point):
            # print(result.array)
            # print(result.crs)
            result = result.to_point(target_geometry)

        return result

    @classmethod
    def merge(
            cls,
            images: List[Raster],
            geometry: RasterGeometry = None,
            crs: CRS = None,
            cell_size: float = None,
            dtype: str = None) -> Raster:
        if geometry is None:
            geometries = [image.geometry for image in images]
            geometry = RasterGrid.merge(geometries=geometries, crs=crs, cell_size=cell_size)

        if dtype is None:
            dtype = images[0].dtype

        if "float" in str(dtype):
            composite_sum = Raster(np.full(geometry.shape, 0, dtype=dtype), geometry=geometry)
            composite_count = Raster(np.full(geometry.shape, 0, dtype=np.uint16), geometry=geometry)

            for image in images:
                projected_image = image.to_geometry(geometry)
                composite_sum = where(np.isnan(projected_image), composite_sum, composite_sum + projected_image)
                projected_tile_count = where(np.isnan(projected_image), 0, 1)
                composite_count = composite_count + projected_tile_count

            composite_image = where(composite_count > 0, composite_sum / composite_count, np.nan)
        else:
            composite_image = Raster(np.full(geometry.shape, 0), geometry=geometry)

            for image in images:
                projected_image = image.to_geometry(geometry)
                composite_image = where(np.isnan(projected_image), composite_image, projected_image)

        return composite_image

    @classmethod
    def from_geotiff(cls, filename: str, **kwargs) -> Raster:
        if not filename.endswith(".tif") or filename.endswith(".tiff"):
            raise ValueError("invalid GeoTIFF filename")

        return cls.from_file(filename, **kwargs)

    # hidden methods

    @staticmethod
    def _raise_unrecognized_key(key):
        raise IndexError(f"key unrecognized: {key}")

    def contain(self, array=None, geometry=None, nodata=None) -> Raster:
        if array is None:
            array = self.array

        if geometry is None:
            geometry = self.geometry

        if np.size(array) == 1:
            return array

        if nodata is None:
            nodata = self.nodata

        return Raster(
            array,
            nodata=nodata,
            metadata=self.metadata,
            cmap=self.cmap,
            geometry=geometry
        )

        # # try:
        # if len(array.shape) == 2:
        #     return Raster(
        #         array,
        #         nodata=nodata,
        #         metadata=self.metadata,
        #         cmap=self.cmap,
        #         geometry=geometry
        #     )
        # elif len(array.shape) == 3:
        #     return MultiRaster(
        #         array,
        #         nodata=nodata,
        #         metadata=self.metadata,
        #         cmap=self.cmap,
        #         geometry=geometry
        #     )
        # else:
        #     raise ValueError(f"invalid raster array with shape {array.shape}")
        # except Exception as e:
        #     return array

    def _contain_func(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if result.shape == self.shape:
                return self.contain(result)
            else:
                return result

        return wrapper

    def _slice_coords(self, y_slice: slice, x_slice: slice) -> Raster:
        return self._slice(*self.geometry._translate_coord_slices(y_slice, x_slice))

    def _slice(self, y_slice: slice, x_slice: slice) -> Raster:
        if any([any([isinstance(index, float) for index in (slice.start, slice.stop, slice.step)]) for slice in
                (y_slice, x_slice)]):
            return self._slice_coords(y_slice, x_slice)

        result = self.array[y_slice, x_slice]
        new_shape = []

        for i, s in enumerate((y_slice, x_slice)):
            indices = s.indices(self.shape[i])
            new_length = indices[1] - indices[0]
            new_shape.append(new_length)

        result.shape = new_shape
        subset_geometry = self.geometry[y_slice, x_slice]
        raster = self.contain(result, geometry=subset_geometry)

        return raster

    def _key(self, key):

        if key is Ellipsis:
            return self.array[...]

        slices = []

        for i in range(2):
            if i > len(key) - 1:
                slices.append(slice(None, None))
            elif isinstance(key[i], int):
                slices.append(slice(key[i], key[i] + 1, None))
            elif isinstance(key[i], slice):
                slices.append(key[i])
            elif key[i] is Ellipsis:
                slices.append(key[i])
            else:
                raise TypeError(f"invalid index: {key[i]}")

        y_slice, x_slice = slices

        return self._slice(y_slice, x_slice)

    def __getitem__(self, key):
        if isinstance(key, (slice, int, tuple)):
            return self._key(key)
        elif isinstance(key, str):
            return self.metadata[key]
        elif isinstance(key, (Raster, np.ndarray)):
            result = self.array[key]
            return result
        else:
            Raster._raise_unrecognized_key(key)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.array[slice] = value
        elif isinstance(key, str):
            self.metadata[key] = value
        elif isinstance(key, (Raster, np.ndarray)):
            self.array[np.array(key)] = value
        else:
            try:
                self.array[np.array(key)] = value
            except Exception as e:
                Raster._raise_unrecognized_key(key)

    def __getattr__(self, item):
        if hasattr(self.array, item):
            result = getattr(self.array, item)

            if callable(result):
                return self._contain_func(result)
            else:
                return self.contain(result)

    @property
    def valid_mask(self) -> Raster:
        data = ~np.logical_or(np.isnan(self.array), self.array == self.nodata)
        mask = self.contain(array=data)

        return mask

    def trim(self, mask: Union[np.ndarray, Raster] = None) -> Raster:
        if mask is None:
            if self.dtype in (np.float32, np.float64):
                mask = ~np.isnan(self.array)
            else:
                mask = self.array.astype(bool)

        mask = ~mask

        first_row, last_row = np.where(~mask.all(axis=1))[0][[0, -1]]
        first_col, last_col = np.where(~mask.all(axis=0))[0][[0, -1]]
        last_row = last_row + 1
        last_col = last_col + 1
        subset = self[first_row:last_row, first_col:last_col]

        return subset

    trimmed = property(trim)

    # read-only properties

    @property
    def nodata(self):
        """
        Nodata value.
        """
        return self._nodata

    @nodata.setter
    def nodata(self, nodata):
        self._nodata = nodata

    @property
    def metadata(self) -> dict:
        """
        Dictionary of metadata.
        """
        return self._metadata

    @property
    def geometry(self) -> RasterGeometry:
        """
        Georeferencing object.
        """
        return self._geometry

    @property
    def crs(self) -> CRS:
        return self.geometry.crs

    @property
    def is_geographic(self) -> bool:
        return self.geometry.is_geographic

    @property
    def proj4(self) -> str:
        return self.geometry.proj4

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Tuple of dimension sizes.
        """
        return self.array.shape

    @property
    def rows(self) -> int:
        """
        Count of rows.
        """
        return self.shape[0]

    @property
    def cols(self) -> int:
        """
        Count of columns.
        """
        return self.shape[1]

    @property
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tuple containing two dimensional array of x coordinates and two dimensional array of y coordinates.
        """
        return self.geometry.xy

    @property
    def latlon(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tuple containing two dimensional array of latitudes and two dimensional array of longitudes.
        """
        return self.geometry.latlon

    @property
    def lat(self) -> np.ndarray:
        """
        Two dimensional array of latitudes.
        """
        return self.geometry.lat

    @property
    def lon(self) -> np.ndarray:
        """
        Two dimensional array of longitudes.
        """
        return self.geometry.lon

    @property
    def x(self) -> np.ndarray:
        """
        Two dimensional array of x coordinates.
        """
        return self.geometry.x

    @property
    def y(self) -> np.ndarray:
        """
        Two dimensional array of y coordinates.
        """
        return self.geometry.y

    @property
    def range(self) -> Tuple[float, float]:
        """
        Minimum and maximum value of the data in a tuple.
        """
        return (np.nanmin(self.array), np.nanmax(self.array))

    @property
    def title(self) -> str:
        """
        Title of the data as a string or None if title not specified. May be constructed from various metadata.
        """
        if 'title' in self.metadata:
            return self.metadata['title']
        elif 'long_name' in self.metadata:
            title = self.metadata['long_name']

            if 'units' in self.metadata:
                title += f" ({self.metadata['units']})"

            return title
        elif 'filename' in self._source_metadata:
            return self._source_metadata['filename']
        else:
            return None

    @property
    def name(self) -> str:
        return self.title

    @property
    def units(self) -> str:
        """
        Units of the data as a string or None if units not specified.
        """
        if 'units' in self.metadata:
            return self.metadata['units']
        else:
            return None

    @property
    def cell_size(self) -> float:
        return self.geometry.cell_size

    @property
    def cell_height(self) -> float:
        return self.geometry.cell_height

    @property
    def cell_width(self) -> float:
        return self.geometry.cell_width

    @property
    def ymin(self) -> float:
        return self.geometry.ymin

    @property
    def xmin(self) -> float:
        return self.geometry.xmin

    @property
    def ymax(self) -> float:
        return self.geometry.ymax

    @property
    def xmax(self) -> float:
        return self.geometry.xmax

    @property
    def x_origin(self) -> float:
        return self.geometry.x_origin

    @property
    def y_origin(self) -> float:
        return self.geometry.y_origin

    @property
    def origin(self) -> float:
        return self.geometry.origin

    @property
    def width(self) -> float:
        return self.geometry.width

    @property
    def height(self) -> float:
        return self.geometry.height

    @property
    def is_gridded(self) -> bool:
        return isinstance(self.geometry, RasterGrid)

    # @profile
    def resample(
            self,
            target_geometry: RasterGeometry,
            nodata: Any = None,
            search_radius_meters: float = None,
            kd_tree: KDTree = None,
            **kwargs) -> Raster:
        """
        This function resamples data from one coordinate field to another by nearest neighbor.
        Either coordinate field may or may not represent gridded data.
        This function should only be used at commensurate scales.
        For disparate scales, use swath_to_swath.
        """
        if nodata is None:
            nodata = self.nodata

        if nodata is np.nan and "int" in str(self.dtype):
            raise ValueError("cannot use NaN as nodata value for integer layer")

        if kd_tree is None:
            kd_tree = KDTree(
                source_geometry=self.geometry,
                target_geometry=target_geometry,
                radius_of_influence=search_radius_meters,
                **kwargs
            )

        output_raster = kd_tree.resample(
            source=self.array,
            fill_value=nodata,
            **kwargs
        )

        return output_raster

    def grid_raster(
            self,
            geometry: RasterGeometry = None,
            crs: Union[CRS, str] = None,
            target_cell_size: float = None,
            kd_tree: KDTree = None,
            **kwargs) -> Raster:
        if isinstance(self.geometry, RasterGrid) and geometry is None and crs is None:
            return self

        if geometry is None:
            geometry = self.geometry.grid_to_crs(
                target_crs=crs,
                target_cell_size=target_cell_size
            )

        image = self.resample(
            target_geometry=geometry,
            kd_tree=kd_tree,
            **kwargs
        )

        return image

    grid = property(grid_raster)

    def resize(self, shape: Tuple[int, int], resampling: str = None):
        return self.to_geometry(self.geometry.resize(shape), resampling=resampling)

    def rescale(self, target_cell_size: float, method: str = 'linear', order: int = None):
        if order is None:
            if method not in SKIMAGE_RESAMPLING_METHODS:
                raise ValueError(f"resampling method not recognized: {method}")

            order = SKIMAGE_RESAMPLING_METHODS[method]

        geometry = self.geometry.reproject(self.crs, target_cell_size)
        data = skimage.transform.resize(
            self.array,
            (self.rows, self.cols),
            order=order,
            preserve_range=True
        )
        raster = self.contain(array=data, geometry=geometry)

        return raster

    @property
    def boundary(self) -> Polygon:
        return self.geometry.boundary

    @property
    def bbox(self) -> BBox:
        return self.geometry.bbox

    @property
    def geolocation(self) -> Raster:
        return self.contain(geometry=self.geometry.geolocation)

    def to_point(self, point: Point, method: str = None, **kwargs):
        # print(f"to_point({point})")
        if isinstance(point, shapely.geometry.Point):
            point = Point(point, crs=WGS84)

        point = point.to_crs(self.crs)
        point._crs = self.crs

        if not self.geometry.intersects(point):
            return np.nan

        if method is None:
            if np.issubdtype(self.dtype, np.integer):
                method = "nearest"
            elif np.issubdtype(self.dtype, np.floating):
                method = "IDW"
            else:
                method = "nearest"

        if method == "nearest":
            index = self.geometry.index_point(point)
            value = self.array[index].item()
        elif method == "IDW":
            value = self.IDW(geometry=point, **kwargs)
        else:
            raise ValueError(f"unrecognized point sampling method: {method}")

        return value

    # @profile
    def to_grid(
            self,
            grid: RasterGrid,
            search_radius_meters = None,
            resampling: str = None,
            kd_tree: KDTree = None,
            nodata: Any = None,
            **kwargs) -> Raster:
        from .CRS import CRS

        if not isinstance(grid, RasterGrid):
            raise TypeError(f"target geometry must be a RasterGrid object, not {type(grid)}")

        if nodata is None:
            nodata = self.nodata

        if nodata is np.nan and "int" in str(self.dtype):
            raise ValueError("cannot use NaN as nodata value for integer layer")

        if resampling is None:
            resampling = "nearest"

        if isinstance(resampling, str) and resampling in RASTERIO_RESAMPLING_METHODS:
            resampling = RASTERIO_RESAMPLING_METHODS[resampling]

        if isinstance(self.geometry, RasterGeolocation):
            if resampling == "nearest":
                return self.resample(
                    target_geometry=grid,
                    search_radius_meters=search_radius_meters,
                    kd_tree=kd_tree
                )
            else:
                return self.resample(target_geometry=self.geometry.grid, search_radius_meters=search_radius_meters, nodata=nodata).to_geometry(grid, resampling=resampling, search_radius_meters=search_radius_meters, nodata=nodata)


        # create source array
        source = np.array(self.array)

        # define affine transforms
        src_transform = self.geometry.affine
        dst_transform = grid.affine

        # define coordinate reference systems
        src_crs = CRS(self.crs)
        dst_crs = CRS(grid.crs)

        destination_dtype = self.dtype

        if nodata is None:
            # define nodata
            if self.dtype in (np.float32, np.float64):
                src_nodata = np.nan
                dst_nodata = np.nan
            elif str(self.dtype) == "bool":
                source = source.astype(np.uint16)
                destination_dtype = np.uint16
                src_nodata = np.uint16(self.nodata)
                dst_nodata = src_nodata
            else:
                src_nodata = self.nodata
                dst_nodata = self.nodata
        else:
            src_nodata = nodata
            dst_nodata = nodata

        # create destination array
        if len(self.shape) == 2:
            destination = np.empty((grid.rows, grid.cols), destination_dtype)
        elif len(self.shape) == 3:
            destination = np.empty((self.shape[0], grid.rows, grid.cols), destination_dtype)
        else:
            raise ValueError(f"invalid raster dimensions: {self.shape}")

        if str(self.dtype) == "bool":
            source = source.astype(np.uint16)

        # resample to destination array
        with rasterio.Env():
            try:
                reproject(
                    source,
                    destination,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                    **kwargs
                )
            except Exception as e:
                self.logger.exception(e)
                raise ValueError(
                    f"unable to project source type {source.dtype} nodata {src_nodata} to type {destination.dtype} nodata {dst_nodata}")

        if self.dtype != destination.dtype:
            destination = destination.astype(self.dtype)

        # encapsulate destination data to raster
        raster = self.contain(destination, geometry=grid)

        return raster

    def to_geolocation(
            self,
            geolocation: RasterGeolocation,
            kd_tree: KDTree = None,
            **kwargs) -> Raster:
        return self.resample(
            target_geometry=geolocation,
            kd_tree=kd_tree,
            **kwargs
        )

    def to_geometry(
            self,
            target_geometry: Union[RasterGeometry, Point],
            resampling: str = None,
            search_radius_meters: float = None,
            kd_tree: KDTree = None,
            nodata: Any = None,
            **kwargs) -> Raster:
        from .point import Point

        if nodata is None:
            nodata = self.nodata

        if nodata is np.nan and "int" in str(self.dtype):
            raise ValueError("cannot use NaN as nodata value for integer layer")

        if self.geometry == target_geometry:
            return self

        if isinstance(target_geometry, RasterGrid):
            return self.to_grid(
                target_geometry,
                search_radius_meters=search_radius_meters,
                resampling=resampling,
                kd_tree=kd_tree,
                nodata=nodata,
                **kwargs
            )
        elif isinstance(target_geometry, RasterGeolocation):
            return self.to_geolocation(
                target_geometry,
                search_radius_meters=search_radius_meters,
                kd_tree=kd_tree,
                **kwargs
            )
        elif isinstance(target_geometry, Point):
            return self.to_point(target_geometry)
        else:
            raise ValueError(f"unsupported target geometry type: {type(target_geometry)}")

    @property
    def count(self) -> int:
        # single-band raster
        return 1

    @property
    def pixel_centroids(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"value": self.array.flatten()}, geometry=list(self.geometry.pixel_centroids))

    @property
    def pixel_outlines(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"value": self.array.flatten()}, geometry=list(self.geometry.pixel_outlines))

    def IDW(self, geometry: VectorGeometry, power: float = 2) -> Union[float, gpd.GeoDataFrame]:
        """
        Performs Inverse Distance Weighting (IDW) interpolation on a given geometry.

        Args:
            geometry (VectorGeometry): The target geometry for interpolation.
            power (float, optional): The power parameter controlling the rate of decay of distance influence. Defaults to 2.

        Returns:
            Union[float, gpd.GeoDataFrame]:
                - If the input geometry is a single point, returns the interpolated value as a float.
                - If the input geometry is a GeoDataFrame, returns a new GeoDataFrame with the interpolated values.

        Raises:
            ValueError: If the input geometry is not supported.
        """
        from .point import Point
        from .vector_geometry import SingleVectorGeometry

        if isinstance(geometry, (Point, shapely.geometry.Point)):
            geometry = wrap_geometry(geometry)

            # Get the centroids of the pixels in the underlying raster
            pixel_centroids = self.geometry.pixel_centroids

            # Calculate distances between the input point and pixel centroids
            distances = geometry.distances(pixel_centroids)

            # Extract values from the raster array
            value = self.array.flatten()

            # Convert distances to a NumPy array
            distance = np.array(distances["distance"])

            # Calculate weights based on inverse distance
            weight = 1 / distance ** power

            # Calculate weighted values
            weighted = value * weight

            # Calculate the interpolated value using weighted summation
            result = np.nansum(weighted) / np.nansum(weight)

            result = result.item()

            # Return the result based on the input geometry type
            if isinstance(geometry, SingleVectorGeometry):
                return result
            else:
                return gpd.GeoDataFrame({"value": [result]}, geometry=[geometry])
        else:
            raise ValueError(f"unsupported target geometry: {geometry}")

    def generate_profile(
            self,
            driver: str = None,
            dtype: str = None,
            nodata: Any = None,
            count: int = None,
            compress: str = DEFAULT_GEOTIFF_COMPRESSION,
            use_compression: bool = True,
            **kwargs) -> dict:
        if driver is None:
            driver = "GTiff"

        if dtype is None:
            if str(self.dtype) == "float64":
                dtype = np.float32
            elif str(self.dtype) == "int64":
                dtype = np.int32
            elif str(self.dtype) == "bool":
                dtype = np.uint16
            else:
                dtype = self.dtype

        if nodata is None:
            nodata = self.nodata

            try:
                if dtype != np.float32 and nodata is np.nan:
                    nodata = None
            except:
                nodata = None

        # using proj4 as a universal CRS encoding is becoming deprecated
        CRS = self.geometry.proj4

        width = self.cols
        height = self.rows
        transform = self.geometry.affine

        if count is None:
            count = self.count

        profile = {
            "count": count,  # singe-band raster
            "width": width,
            "height": height,
            "nodata": nodata,
            "dtype": dtype,
            "crs": CRS,
            "transform": transform,
            "driver": driver
        }

        profile.update(kwargs)

        for key in list(profile.keys()):
            if profile[key] is None:
                del (profile[key])

        if use_compression and compress is not None:
            profile["compress"] = compress

        return profile

    rasterio_profile = property(generate_profile)

    def to_file(
            self,
            filename: str,
            driver: str = None,
            compress: str = None,
            use_compression: bool = True,
            overwrite: bool = True,
            nodata: Any = None,
            **kwargs):
        if not isinstance(filename, str):
            raise ValueError("invalid filename")

        filename = abspath(expanduser(filename))

        if driver is None:
            driver = GEOTIFF_DRIVER

        if compress is None:
            compress = self.DEFAULT_GEOTIFF_COMPRESSION

        profile = self.generate_profile(
            driver=driver,
            compress=compress,
            use_compression=use_compression,
            nodata=nodata,
            **kwargs
        )

        filename = expanduser(filename)
        filename = abspath(filename)
        self.metadata["filename"] = filename
        dtype = profile["dtype"]
        directory = dirname(filename)
        output_array = self.array.astype(dtype)

        if not exists(directory) and not directory == '':
            makedirs(directory, exist_ok=True)

        if exists(filename) and not overwrite:
            raise IOError(f"output file already exists: {filename}")

        if len(self.shape) == 2:
            with rasterio.open(filename, "w", **profile) as file:
                file.write(output_array, 1)
        else:
            with rasterio.open(filename, "w", **profile) as file:
                file.write(output_array)

    def to_geotiff(
            self,
            filename: str,
            compression: str = None,
            preview_quality: int = 20,
            cmap: Union[Colormap, str] = None,
            use_compression: bool = True,
            include_preview: bool = True,
            overwrite: bool = True,
            **kwargs):
        self.to_file(
            filename=filename,
            driver=GEOTIFF_DRIVER,
            compress=compression,
            use_compression=use_compression,
            **kwargs
        )

        if include_preview:
            self.to_geojpeg(
                filename=f"{splitext(filename)[0]}.jpeg",
                cmap=cmap,
                quality=preview_quality,
                overwrite=overwrite
            )

    def to_geopackage(
            self,
            filename: str,
            compression: str = None,
            preview_quality: int = 20,
            cmap: Union[Colormap, str] = DEFAULT_CMAP,
            use_compression: bool = True,
            include_preview: bool = True,
            overwrite: bool = True,
            **kwargs):
        self.to_file(
            filename=filename,
            driver=GEOPACKAGE_DRIVER,
            compress=compression,
            use_compression=use_compression,
            **kwargs
        )

        if include_preview:
            self.to_geojpeg(
                filename=f"{splitext(filename)[0]}.jpeg",
                cmap=cmap,
                quality=preview_quality,
                overwrite=overwrite
                )

    def to_COG(
            self,
            filename: str,
            compression: str = "deflate",
            preview_quality: int = 20,
            cmap: Union[Colormap, str] = DEFAULT_CMAP,
            include_preview: bool = True,
            overwrite: bool = True,
            remove_XML: bool = True,
            **kwargs):
        filename = abspath(expanduser(filename))
        temporary_filename = filename.replace(".tif", ".temp.tif")

        if exists(temporary_filename):
            os.remove(temporary_filename)

        # print(f"writing temporary GeoTIFF: {temporary_filename}")

        self.to_geotiff(
            filename=temporary_filename,
            include_preview=False,
            overwrite=True
        )

        if not exists(temporary_filename):
            raise IOError(f"unable to create temporary GeoTIFF: {temporary_filename}")

        # print(f"temporary file exists: {exists(temporary_filename)}")

        command = f'gdal_translate "{temporary_filename}" "{filename}" -co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co COMPRESS={compression.upper()}'
        # print(command)
        os.system(command)

        # print(f"final file exists: {exists(filename)}")

        os.remove(temporary_filename)

        XML_filename = f"{filename}.aux.xml"

        if remove_XML and exists(XML_filename):
            os.remove(XML_filename)

        if include_preview:
            self.to_geojpeg(
                filename=f"{splitext(filename)[0]}.jpeg",
                cmap=cmap,
                quality=preview_quality,
                overwrite=overwrite
            )

    def to_geojpeg(
            self,
            filename: str,
            cmap: Union[Colormap, str] = None,
            quality: int = 75,
            overwrite: bool = False,
            **kwargs):
        if not isinstance(filename, str):
            raise ValueError("invalid filename")

        filename = abspath(expanduser(filename))

        if exists(filename) and not overwrite:
            # self.logger.info(f"GeoJPEG file already exists: {filename}")
            return

        rendered_image = self.to_pillow(cmap=cmap)
        rendered_array = np.array(rendered_image)
        reshaped_array = np.moveaxis(rendered_array, -1, 0)

        profile = self.generate_profile(
            quality=quality,
            compress=None,
            driver=GEOJPEG_DRIVER,
            dtype="uint8",
            count=3,
            **kwargs
        )

        makedirs(dirname(abspath(filename)), exist_ok=True)

        with rasterio.Env(CPL_LOG_ERRORS="OFF"):
            with rasterio.open(filename, "w", **profile) as file:
                file.write(reshaped_array)

        if not exists(filename):
            raise IOError(f"unable to create GeoJPEG file: {filename}")

    def to_geopng(
            self,
            filename: str,
            cmap: Union[Colormap, str] = None,
            quality: int = 75,
            overwrite: bool = False,
            **kwargs):
        if not isinstance(filename, str):
            raise ValueError("invalid filename")

        filename = abspath(expanduser(filename))

        if exists(filename) and not overwrite:
            # self.logger.info(f"GeoJPEG file already exists: {filename}")
            return

        rendered_image = self.to_pillow(cmap=cmap)
        rendered_array = np.array(rendered_image)
        reshaped_array = np.moveaxis(rendered_array, -1, 0)

        profile = self.generate_profile(
            quality=quality,
            compress=None,
            driver=GEOPNG_DRIVER,
            dtype="uint8",
            count=3,
            **kwargs
        )

        makedirs(dirname(abspath(filename)), exist_ok=True)

        with rasterio.Env(CPL_LOG_ERRORS="OFF"):
            with rasterio.open(filename, "w", **profile) as file:
                file.write(reshaped_array)

        if not exists(filename):
            raise IOError(f"unable to create GeoPNG file: {filename}")

    @property
    def is_float(self):
        return str(self.dtype).startswith("f")

    @property
    def empty(self):
        if self.is_float:
            return np.all(np.isnan(self.array))
        else:
            return np.all(self.array == self.nodata)

    def reproject(
            self,
            crs: Union[CRS, str] = WGS84,
            target_cell_size: float = None,
            method: str = "nearest",
            kd_tree: KDTree = None):
        if method not in RASTERIO_RESAMPLING_METHODS:
            raise ValueError(f"resampling method is not supported: {method}")

        # warp target georeference
        destination_geometry = self.geometry.reproject(
            crs=crs,
            cell_size=target_cell_size
        )

        # select rasterio resampling method
        resampling = RASTERIO_RESAMPLING_METHODS[method]

        if self.is_gridded:
            raster = self.to_grid(
                grid=destination_geometry,
                resampling=resampling,
                kd_tree=kd_tree
            )
        else:
            raster = self.resample(
                target_geometry=destination_geometry,
                kd_tree=kd_tree
            )

        return raster

    def UTM(self, target_cell_size: float, kd_tree: KDTree = None) -> Raster:
        return self.reproject(
            crs=self.geometry.local_UTM_proj4,
            target_cell_size=target_cell_size,
            kd_tree=kd_tree
        )

    def mask(self, mask: Union[Raster, np.ndarray]) -> Raster:
        return where(mask, self, np.nan)

    def fill(self, other: Union[Raster, np.ndarray]) -> Raster:
        if self.shape != other.shape:
            raise ValueError(f"raster with shape {self.shape} cannot be filled with raster of shape {other.shape}")

        return where(np.isnan(self), other, self)

    def imshow(
            self,
            title: str = None,
            style: str = None,
            cmap: Union[str, Colormap] = None,
            figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
            facecolor: str = None,
            vmin: float = None,
            vmax: float = None,
            fig=None,
            ax=None,
            backend: str = "Agg",
            hide_ticks: bool = False,
            render: bool = True,
            diverging: bool = False,
            **kwargs) -> Union[Figure, Image]:
        prior_backend = plt.get_backend()
        plt.switch_backend(backend)

        plt.close()

        if style is None:
            style = DEFAULT_MATPLOTLIB_STYLE

        if facecolor is None:
            if "dark" in style:
                facecolor = "black"
            else:
                facecolor = "white"
        
        if cmap is None:
            cmap = self.cmap

        with plt.style.context(style):
            if fig is None or ax is None:
                fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=facecolor, figsize=figsize)

            if title is None:
                title = self.title

            if title is not None:
                ax.set_title(title)

            data = self.array

            tick_labels = None

            if str(self.array.dtype) == "bool":
                data = self.array.astype(np.uint8)
                boundaries = [0, 1]
                ticks = [0, 1]
                tick_labels = ["False", "True"]
                vmin = 0
                vmax = 1

                if cmap is None:
                    cmap = colors.ListedColormap(["black", "white"])
            elif np.issubdtype(self.array.dtype, np.integer) and list(np.unique(data)) == [0, 1]:
                boundaries = [0, 1]
                ticks = [0, 1]
                vmin = 0
                vmax = 1

                if cmap is None:
                    cmap = colors.ListedColormap(["black", "white"])
            elif len(np.unique(data)) < 10:
                vmin = None
                vmax = None
                boundaries = None
                ticks = None
            else:
                if vmin is None and vmax is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        minimum = np.nanmin(data)
                        maximum = np.nanmax(data)
                        mean = np.nanmean(data)
                        sd = np.nanstd(data)

                    vmin = max(mean - 2 * sd, minimum)
                    vmax = min(mean + 2 * sd, maximum)

                    if diverging:
                        vradius = max(abs(vmin), abs(vmax))
                        vmin = -vradius
                        vmax = vradius

                boundaries = None
                ticks = None

            if cmap is None:
                if self.cmap is None:
                    cmap = "jet"
                else:
                    cmap = self.cmap

            im = ax.imshow(
                data,
                extent=self.geometry._matplotlib_extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            if self.is_geographic:
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')

                units = u'\N{DEGREE SIGN}'
            else:
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')

                if "units=m" in self.proj4:
                    units = 'm'
                else:
                    units = ''

            if "units=m" in self.proj4:
                def format_tick_meters(tick_value, position):
                    if abs(tick_value) > 1000:
                        tick_value /= 1000
                        units = "km"
                    else:
                        units = "m"

                    tick_string = f"{tick_value:0.2f}"

                    if tick_string.endswith(".00"):
                        tick_string = tick_string[:-3]

                    return f"{tick_string} {units}"

                tick_formatter = FuncFormatter(format_tick_meters)
            else:
                def format_tick(tick_value, position):
                    tick_string = f"{tick_value:0.2f}"

                    if tick_string.endswith(".00"):
                        tick_string = tick_string[:-3]

                    return f"{tick_string} {units}"

                tick_formatter = FuncFormatter(format_tick)

            ax.get_xaxis().set_major_formatter(tick_formatter)
            ax.get_yaxis().set_major_formatter(tick_formatter)
            plt.xticks(rotation=-90)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, boundaries=boundaries, ticks=ticks)

            if tick_labels is not None:
                cbar.ax.set_yticklabels(tick_labels)
                cbar.ax.tick_params(labelsize=14)

            if hide_ticks:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            fig.tight_layout()

        plt.switch_backend(prior_backend)

        if render:
            output = self.image(fig=fig, **kwargs)
        else:
            output = fig

        return output

    def render(
            self,
            fig=None,
            format: str = "png",
            transparent: bool = False,
            style: str = DEFAULT_MATPLOTLIB_STYLE,
            dpi: int = DEFAULT_DPI,
            **kwargs) -> bytes:
        if fig is None:
            fig = self.imshow(style=style, render=False, **kwargs)

        with plt.style.context(style):
            buffer = io.BytesIO()
            fig.savefig(buffer, format=format, transparent=transparent, dpi=dpi)
            plt.close(fig)

        buffer.seek(0)

        return buffer

    def image(
            self,
            fig=None,
            format: str = "png",
            transparent: bool = False,
            style: str = DEFAULT_MATPLOTLIB_STYLE,
            dpi: int = DEFAULT_DPI,
            **kwargs) -> Image:
        buffer = self.render(fig=fig, format=format, transparent=transparent, style=style, dpi=dpi, **kwargs)
        image = PIL.Image.open(buffer)

        return image

    def _repr_png_(self) -> bytes:
        return self.render(format='png').getvalue()

    def __str__(self):
        return self.array.__str__()

    @property
    def minmaxstretch(self) -> Raster:
        if np.all(np.isnan(self.array)):
            return self

        transform = MinMaxInterval()
        stretch = transform(self.array)

        return self.contain(stretch)

    def generate_percentilecut(self, lower_percentile=2, upper_percentile=98) -> Raster:
        if np.all(np.isnan(self.array)):
            return self

        transform = AsymmetricPercentileInterval(lower_percentile, upper_percentile)
        stretch = transform(self.array)

        return self.contain(stretch)

    percentilecut = property(generate_percentilecut)

    def to_pillow(
            self,
            cmap: Union[Colormap, str] = None,
            mode: str = "RGB") -> PIL.Image.Image:
        DEFAULT_CONTINUOUS_CMAP = plt.get_cmap("jet")
        DEFAULT_BINARY_CMAP = colors.ListedColormap(["black", "white"])

        if cmap is None:
            cmap = self.cmap
        
        if str(self.array.dtype) == "bool":
            # data = self.array.astype(np.uint8)
            vmin = 0
            vmax = 1

            if cmap is None:
                cmap = colors.ListedColormap(["black", "white"])

        elif np.issubdtype(self.array.dtype, np.integer) and np.all(tuple(np.unique(self.array)) == (0, 1)):
            # data = self.array.astype(np.uint8)
            vmin = 0
            vmax = 1

            if cmap is None:
                cmap = colors.ListedColormap(["black", "white"])
        elif len(np.unique(self.array)) < 10:
            vmin = None
            vmax = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                minimum = np.nanmin(self.array)
                maximum = np.nanmax(self.array)
                mean = np.nanmean(self.array)
                sd = np.nanstd(self.array)

            vmin = max(mean - 2 * sd, minimum)
            vmax = min(mean + 2 * sd, maximum)

        if cmap is None:
            if self.cmap is None:
                cmap = "jet"
            else:
                cmap = self.cmap

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if len(np.unique(self.array)) == 1:
            image_array_norm = np.full(self.shape, 0)
        elif vmin is None and vmax is None:
            image_array_norm = cmap(np.array(self.minmaxstretch))
        else:
            image_array_norm = cmap(np.array(self.clip(vmin, vmax).minmaxstretch))

        image_array_int = np.uint8(image_array_norm * 255)
        pillow_image = PIL.Image.fromarray(image_array_int)

        if mode is not None:
            pillow_image = pillow_image.convert(mode)

        return pillow_image

    pillow = property(to_pillow)
