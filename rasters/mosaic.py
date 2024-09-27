from __future__ import annotations

from typing import Iterator, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .raster import Raster
    from .raster_geometry import RasterGeometry

def mosaic(images: Iterator[Union[Raster, str]], geometry: RasterGeometry) -> Raster:
    from .where import where

    mosaic = Raster(np.full(geometry.shape, np.nan), geometry=geometry)
    dtype = None
    nodata = None
    metadata = None

    for image in images:
        if isinstance(image, str):
            image = Raster.open(image)

        dtype = image.dtype
        nodata = image.nodata
        metadata = image.metadata
        mosaic = where(np.isnan(mosaic), image.to_geometry(geometry), mosaic)

    mosaic = mosaic.astype(dtype)
    mosaic = Raster(mosaic, geometry=geometry, nodata=nodata, metadata=metadata)

    return mosaic
