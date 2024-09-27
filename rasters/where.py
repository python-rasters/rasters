from __future__ import annotations

from typing import Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .raster import Raster

def where(condition, x, y):
    from .raster import Raster

    if isinstance(condition, Raster):
        geometry = condition.geometry
    elif isinstance(x, Raster):
        geometry = x.geometry
    elif isinstance(y, Raster):
        geometry = y.geometry
    else:
        geometry = None

    cmap = None

    if hasattr(x, "cmap") and getattr(x, "cmap") is not None:
        cmap = x.cmap

    if cmap is None and hasattr(y, "cmap") and getattr(y, "cmap") is not None:
        cmap = y.cmap

    metadata = {}

    if hasattr(x, "metadata") and len(getattr(x, "metadata")) > 0:
        for key, value in x.metadata.items():
            if key not in metadata:
                metadata[key] = value

    if hasattr(y, "metadata") and len(getattr(y, "metadata")) > 0:
        for key, value in y.metadata.items():
            if key not in metadata:
                metadata[key] = value

    nodata = None

    if hasattr(x, "nodata") and getattr(x, "nodata") is not None:
        nodata = x.nodata

    if nodata is None and hasattr(y, "nodata") and getattr(y, "nodata") is not None:
        nodata = y.nodata

    result = np.where(condition, x, y)

    if geometry is None:
        return result
    else:
        return Raster(result, geometry=geometry, cmap=cmap, metadata=metadata, nodata=nodata)
