from __future__ import annotations

from typing import Union, Tuple, List, TYPE_CHECKING

import PIL.Image
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .constants import *
from .raster import Raster

if TYPE_CHECKING:
    from .raster_geometry import RasterGeometry
    from .raster import Raster

class MultiRaster(Raster):
    multi = True

    def __init__(
            self,
            array: Union[np.ndarray, Raster],
            geometry: RasterGeometry,
            nodata=None,
            cmap=None,
            **kwargs):
        from .raster import Raster
        from .raster_geometry import RasterGeometry

        ALLOWED_ARRAY_TYPES = (
            np.ndarray,
            h5py.Dataset
        )

        if isinstance(array, Raster):
            array = array.array

        if not isinstance(array, ALLOWED_ARRAY_TYPES):
            raise TypeError('data is not a valid numpy.ndarray')

        ndim = len(array.shape)

        if ndim == 2:
            rows, cols = array.shape
            array = np.reshape(array, (1, rows, cols))
        elif ndim != 3:
            raise ValueError('data is not a three-dimensional array')

        dtype = array.dtype

        if dtype in (np.float32, np.float64):
            array = np.where(array == nodata, np.nan, array)

        self._array = array

        if nodata is None:
            if dtype in (np.float32, np.float64):
                nodata = np.nan

        self._nodata = nodata
        self._metadata = {}

        for key, value in kwargs.items():
            self._metadata[key] = value

        if isinstance(geometry, RasterGeometry):
            self._geometry = geometry
        else:
            raise ValueError(f"geometry is not a valid RasterGeometry object: {type(geometry)}")

        self.cmap = cmap
        self._source_metadata = {}

    def contain(self, array=None, geometry=None, nodata=None) -> Raster:
        if array is None:
            array = self.array

        if geometry is None:
            geometry = self.geometry

        if np.size(array) == 1:
            return array

        if nodata is None:
            nodata = self.nodata

        return MultiRaster(
            array,
            nodata=nodata,
            metadata=self.metadata,
            cmap=self.cmap,
            geometry=geometry
        )
        
    @classmethod
    def stack(cls, rasters: List[Raster], *args, **kwargs) -> MultiRaster:
        geometry = rasters[0].geometry
        stack = np.stack(rasters)
        image = MultiRaster(stack, *args, geometry=geometry, **kwargs)

        return image

    def band(self, band: int) -> Raster:
        from .raster import Raster
        
        image = Raster.contain(self, self.array[band, ...])
        return image

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
        if self.count == 1:
            return self.band(0).imshow(
                title=title,
                style=style,
                cmap=cmap,
                figsize=figsize,
                facecolor=facecolor,
                vmin=vmin,
                vmax=vmax,
                fig=fig,
                ax=ax,
                hide_ticks=hide_ticks
            )

        plt.close()

        if style is None:
            style = DEFAULT_MATPLOTLIB_STYLE

        if facecolor is None:
            if "dark" in style:
                facecolor = "black"
            else:
                facecolor = "white"

        with plt.style.context(style):
            if fig is None or ax is None:
                fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=facecolor, figsize=figsize)

            if title is None:
                title = self.title

            if title is not None:
                ax.set_title(title)

            data = self.array

            tick_labels = None

            red = self.band(0).percentilecut
            green = self.band(1).percentilecut
            blue = self.band(2).percentilecut

            array = np.dstack([red, green, blue])

            im = ax.imshow(
                array,
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

            if hide_ticks:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            plt.tight_layout()

            if render:
                output = self.image(fig=fig, **kwargs)
            else:
                output = fig

            return output

    def to_pillow(
            self,
            cmap: Union[Colormap, str] = None,
            mode: str = "RGB") -> PIL.Image.Image:
        if self.count == 1:
            return self.band(0).to_pillow(cmap=cmap, mode=mode)

        red = self.band(0).percentilecut
        green = self.band(1).percentilecut
        blue = self.band(2).percentilecut

        pillow_image = PIL.Image.fromarray(np.uint8(np.stack([red, green, blue], axis=2) * 255))

        if mode is not None:
            pillow_image = pillow_image.convert(mode)

        return pillow_image

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple of dimension sizes.
        """
        return self.array.shape

    @property
    def count(self) -> int:
        return self.shape[0]

    @property
    def rows(self) -> int:
        """
        Count of rows.
        """
        return self.shape[1]

    @property
    def cols(self) -> int:
        """
        Count of columns.
        """
        return self.shape[2]

    def generate_percentilecut(self, lower_percentile=2, upper_percentile=98) -> Raster:
        return MultiRaster.stack([
            self.band(band).generate_percentilecut(lower_percentile=lower_percentile, upper_percentile=upper_percentile)
            for band
            in range(self.count)
        ])

    def resize(self, shape: Tuple[int, int], resampling: str = None):
        return MultiRaster.stack([
            self.band(band).resize(shape=shape, resampling=resampling)
            for band
            in range(self.count)
        ])
