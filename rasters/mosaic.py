from __future__ import annotations
import warnings
from typing import Iterator, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .raster import Raster
    from .raster_geometry import RasterGeometry

def mosaic(
        images: Iterator[Union[Raster, str]], 
        geometry: RasterGeometry,
        resampling: str = "nearest") -> Raster:
    """
    Creates a mosaic from a sequence of Raster images.

    This function takes an iterator of Raster objects or file paths to Raster files
    and combines them into a single mosaic image with the specified geometry.
    The mosaic is created by overlaying the images in the order they are provided,
    with later images taking precedence over earlier ones.

    Args:
        images (Iterator[Union[Raster, str]]): An iterator of Raster objects or 
                                                file paths to Raster files.
        geometry (RasterGeometry): The desired geometry for the output mosaic.

    Returns:
        Raster: The mosaic image as a Raster object.
    """
    from .where import where  # Assuming 'where' is a function in the same package
    from .raster import Raster  # Import Raster here to avoid circular dependency

    mosaic = Raster(np.full(geometry.shape, np.nan), geometry=geometry)  # Initialize with NaN values
    dtype = None
    nodata = None
    metadata = None

    for image in images:
        if isinstance(image, str):
            image = Raster.open(image)  # Open the image if it's a file path

        dtype = image.dtype
        nodata = image.nodata
        metadata = image.metadata
        
        # Overlay the image onto the mosaic using the 'where' function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mosaic = where(np.isnan(mosaic), image.to_geometry(geometry, resampling=resampling), mosaic)

    mosaic = mosaic.astype(dtype)  # Set the data type of the mosaic
    mosaic = Raster(mosaic, geometry=geometry, nodata=nodata, metadata=metadata)  # Create the final Raster

    return mosaic
