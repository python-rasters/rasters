from .raster import Raster
from .where import where

DEFAULT_UPSAMPLING = "average"
DEFAULT_DOWNSAMPLING = "linear"

def bias_correct(
        coarse_image: Raster,
        fine_image: Raster,
        upsampling: str = "average",
        downsampling: str = "linear",
        return_bias: bool = False):
    fine_geometry = fine_image.geometry
    coarse_geometry = coarse_image.geometry
    upsampled = fine_image.to_geometry(coarse_geometry, resampling=upsampling)
    bias_coarse = upsampled - coarse_image
    bias_fine = bias_coarse.to_geometry(fine_geometry, resampling=downsampling)
    bias_corrected_fine = fine_image - bias_fine

    if return_bias:
        return bias_corrected_fine, bias_fine
    else:
        return bias_corrected_fine
