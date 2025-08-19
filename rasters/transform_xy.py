from pyproj import Transformer
import numpy as np

def transform_xy(
    x: np.ndarray,
    y: np.ndarray,
    source_crs,
    target_crs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform arrays of x and y coordinates from a source CRS to a target CRS.

    Parameters
    ----------
    x : np.ndarray
        Array of x coordinates (e.g., longitude or easting).
    y : np.ndarray
        Array of y coordinates (e.g., latitude or northing).
    source_crs : pyproj.CRS, str, or custom CRS
        The source coordinate reference system. Can be a pyproj.CRS, proj4 string, EPSG code, or custom CRS class.
    target_crs : pyproj.CRS, str, or custom CRS
        The target coordinate reference system. Can be a pyproj.CRS, proj4 string, EPSG code, or custom CRS class.

    Returns
    -------
    x_t : np.ndarray
        Transformed x coordinates in the target CRS.
    y_t : np.ndarray
        Transformed y coordinates in the target CRS.

    Notes
    -----
    - Handles both geographic and projected CRS.
    - If the target CRS is geographic, output coordinates outside valid bounds are set to np.nan.
    - Accepts pyproj.CRS, string, or custom CRS class with 'is_geographic' attribute.
    """
    try:
        from .CRS import CRS
        if not isinstance(source_crs, CRS):
            source_crs = CRS(source_crs)
        if not isinstance(target_crs, CRS):
            target_crs = CRS(target_crs)
    except ImportError:
        pass

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x_t, y_t = transformer.transform(x, y)

    # If target CRS is geographic, clip to valid bounds
    if hasattr(target_crs, 'is_geographic') and target_crs.is_geographic:
        x_t = np.where(np.logical_or(x_t < -180, x_t > 180), np.nan, x_t)
        y_t = np.where(np.logical_or(y_t < -90, y_t > 90), np.nan, y_t)

    return x_t, y_t
