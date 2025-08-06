import os

def test_open_geotiff():
    """Test opening a GeoTIFF file using rasterio."""
    geotiff_path = os.path.join(os.path.dirname(__file__), "test_input.tif")
    from rasters.raster import Raster
    raster = Raster.open(geotiff_path)
    assert raster.count > 0, "GeoTIFF should have at least one band"
    assert raster.width > 0 and raster.height > 0, "GeoTIFF should have valid dimensions"
