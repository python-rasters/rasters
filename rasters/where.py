from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .raster import Raster

def where(condition, x, y):
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    This is a wrapper around `numpy.where` that supports Raster objects.
    If any of the inputs are Raster objects, the output will be a Raster object
    with the same geometry, cmap, metadata, and nodata as the input Raster.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : ndarray or Raster
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere. If any of the inputs are Raster objects, the output
        will be a Raster object.

    See Also
    --------
    numpy.where : Equivalent function in NumPy.

    Examples
    --------
    >>> a = np.arange(10)
    >>> where(a < 5, a, 10*a)
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    >>> from .raster import Raster
    >>> r1 = Raster(np.arange(10).reshape(2, 5))
    >>> r2 = Raster(np.arange(10, 20).reshape(2, 5))
    >>> where(r1 < 5, r1, r2)
    Raster([[ 0,  1,  2,  3,  4],
            [10, 11, 12, 13, 14]])
    """
    from .raster import Raster

    # Determine the geometry of the output Raster (if any)
    geometry = None
    if isinstance(condition, Raster):
        geometry = condition.geometry
    elif isinstance(x, Raster):
        geometry = x.geometry
    elif isinstance(y, Raster):
        geometry = y.geometry

    # Determine the cmap of the output Raster (if any)
    cmap = None
    if hasattr(x, "cmap") and getattr(x, "cmap") is not None:
        cmap = x.cmap
    if cmap is None and hasattr(y, "cmap") and getattr(y, "cmap") is not None:
        cmap = y.cmap

    # Determine the metadata of the output Raster (if any)
    metadata = {}
    if hasattr(x, "metadata") and len(getattr(x, "metadata")) > 0:
        for key, value in x.metadata.items():
            if key not in metadata:
                metadata[key] = value
    if hasattr(y, "metadata") and len(getattr(y, "metadata")) > 0:
        for key, value in y.metadata.items():
            if key not in metadata:
                metadata[key] = value

    # Determine the nodata value of the output Raster (if any)
    nodata = None
    if hasattr(x, "nodata") and getattr(x, "nodata") is not None:
        nodata = x.nodata
    if nodata is None and hasattr(y, "nodata") and getattr(y, "nodata") is not None:
        nodata = y.nodata

    # Perform the actual where operation
    result = np.where(condition, x, y)

    # Return the result as a Raster if any of the inputs were Rasters
    if geometry is None:
        return result
    else:
        return Raster(result, geometry=geometry, cmap=cmap, metadata=metadata, nodata=nodata)
