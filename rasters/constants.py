from rasterio.warp import Resampling

DEFAULT_MATPLOTLIB_STYLE = "dark_background"

CELL_SIZE_TO_SEARCH_DISTANCE_FACTOR = 3

GEOTIFF_DRIVER = "GTiff"
GEOPACKAGE_DRIVER = "GPKG"
GEOPNG_DRIVER = "PNG"
GEOJPEG_DRIVER = "JPEG"
COG_DRIVER = "COG"

RASTERIO_RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "linear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "gauss": Resampling.gauss,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3
}

SKIMAGE_RESAMPLING_METHODS = {
    "nearest": 0,
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "quartic": 4,
    "quintic": 5
}

DEFAULT_CMAP = "jet"
DEFAULT_FIGSIZE = (7, 5)
DEFAULT_DPI = 200

DEFAULT_ASCII_SHAPE = (15, 50)
DEFAULT_ASCII_RAMP = "@%#*+=-:. "[::-1]
DEFAULT_NODATA_CHARACTER = " "
