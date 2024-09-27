from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .raster import Raster

def clip(a: Union[Raster, np.ndarray], a_min, a_max, out=None, **kwargs) -> Union[Raster, np.ndarray]:
    from .raster import Raster

    if a_min is None and a_max is None:
        return a

    # result = np.clip(a=a, a_min=a_min, a_max=a_max, out=out, **kwargs)
    result = a

    if a_min is not None:
        result = np.where(result < a_min, a_min, result)
    if a_max is not None:
        result = np.where(result > a_max, a_max, result)

    if isinstance(a, Raster):
        result = a.contain(result)
        
    return result
