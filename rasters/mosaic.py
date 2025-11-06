from __future__ import annotations
import warnings
from typing import Iterator, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .raster import Raster
    from .multi_raster import MultiRaster
    from .raster_geometry import RasterGeometry

def mosaic(
        images: Iterator[Union[Raster, MultiRaster, str]], 
        geometry: RasterGeometry = None,
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
    from .multi_raster import MultiRaster  # Import MultiRaster here to avoid circular dependency

    if len(images) == 0:
        raise ValueError("No images provided for mosaicking.")

    if len(images[0].shape == 2):
        mosaic = Raster(np.full(geometry.shape, np.nan), geometry=geometry)
    elif len(images[0].shape == 3):
        mosaic = MultiRaster(np.full(geometry.shape, np.nan), geometry=geometry)
    else:
        raise ValueError("Unsupported image shape for mosaicking.")
    
    dtype = None
    nodata = None
    metadata = None

    if geometry is None:
        geometry = images[0].geometry

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
    
    if len(mosaic.shape) == 2:
        mosaic = Raster(mosaic, geometry=geometry, nodata=nodata, metadata=metadata)  # Create the final Raster
    elif len(mosaic.shape) == 3:
        mosaic = MultiRaster(mosaic, geometry=geometry, nodata=nodata, metadata=metadata)  # Create the final MultiRaster
    else:
        raise ValueError("Unsupported mosaic shape after processing.")

    return mosaic
