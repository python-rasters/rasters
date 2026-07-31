from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np
from geopandas.array import GeometryArray  # Import GeometryArray for type checking
from shapely.geometry import box  # Import for bounding box clipping

if TYPE_CHECKING:
    from .raster import Raster

def clip(a: Union[Raster, np.ndarray], a_min, a_max, out=None, **kwargs) -> Union[Raster, np.ndarray]:
    """
    Clips the values of an array, Raster, or GeometryArray to a specified range.

    This function limits the values in the input array, Raster, or GeometryArray to fall within the 
    range defined by `a_min` and `a_max`. For arrays, values below `a_min` are set to `a_min`, 
    and values above `a_max` are set to `a_max`. For GeometryArray, geometries are clipped to 
    the bounding box defined by `a_min` and `a_max`.

    Args:
        a (Union[Raster, np.ndarray, GeometryArray]): The input data to clip.
        a_min (float, optional): The minimum value or bounding box limit. If None, no minimum clipping is performed.
        a_max (float, optional): The maximum value or bounding box limit. If None, no maximum clipping is performed.
        out (np.ndarray, optional): An optional output array to store the result. 
                                   Must be the same shape and dtype as `a`.
        **kwargs: Additional keyword arguments passed to np.clip (currently unused).

    Returns:
        Union[Raster, np.ndarray, GeometryArray]: The clipped data. If `a` is a Raster, a new 
                                   Raster object with the clipped data is returned. 
                                   Otherwise, a NumPy array or GeometryArray is returned.
    """
    from .raster import Raster  # Import here to avoid circular dependency

    if a_min is None and a_max is None:
        return a

    result = a  # Initialize result with the input data

    if isinstance(result, GeometryArray):
        if a_min is None or a_max is None:
            raise ValueError("Both a_min and a_max must be specified for GeometryArray clipping.")

        # Create a bounding box from a_min and a_max
        bbox = box(a_min[0], a_min[1], a_max[0], a_max[1])
        
        # Clip each geometry to the bounding box
        result = GeometryArray([geom.intersection(bbox) for geom in result])
        return result

    if a_min is not None:
        result = np.where(result < a_min, a_min, result)  # Clip values below a_min
    if a_max is not None:
        result = np.where(result > a_max, a_max, result)  # Clip values above a_max

    if isinstance(a, Raster):
        result = a.contain(result)  # Ensure the result is contained within the original Raster
        
    return result
