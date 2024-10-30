from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .raster import Raster

def clip(a: Union[Raster, np.ndarray], a_min, a_max, out=None, **kwargs) -> Union[Raster, np.ndarray]:
    """
    Clips the values of an array or Raster to a specified range.

    This function limits the values in the input array or Raster to fall within the 
    range defined by `a_min` and `a_max`. Values below `a_min` are set to `a_min`, 
    and values above `a_max` are set to `a_max`.

    Args:
        a (Union[Raster, np.ndarray]): The input array or Raster to clip.
        a_min (float, optional): The minimum value to clip to. If None, no minimum clipping is performed.
        a_max (float, optional): The maximum value to clip to. If None, no maximum clipping is performed.
        out (np.ndarray, optional): An optional output array to store the result. 
                                   Must be the same shape and dtype as `a`.
        **kwargs: Additional keyword arguments passed to np.clip (currently unused).

    Returns:
        Union[Raster, np.ndarray]: The clipped array or Raster. If `a` is a Raster, a new 
                                   Raster object with the clipped data is returned. 
                                   Otherwise, a NumPy array is returned.
    """
    from .raster import Raster  # Import here to avoid circular dependency

    if a_min is None and a_max is None:
        return a

    result = a  # Initialize result with the input data

    if a_min is not None:
        result = np.where(result < a_min, a_min, result)  # Clip values below a_min
    if a_max is not None:
        result = np.where(result > a_max, a_max, result)  # Clip values above a_max

    if isinstance(a, Raster):
        result = a.contain(result)  # Ensure the result is contained within the original Raster
        
    return result
