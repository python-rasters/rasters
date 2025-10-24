import numpy as np
from rasters import Raster
from rasters import Point

# reading URI: netcdf:"/Users/halverso/data/GEOS5FP/2021.05.29/GEOS.fp.asm.tavg3_2d_aer_Nx.20210529_1930.V01.nc4":TOTEXTTAU nodata: nan geometry: POINT (-110.0522 31.7438) resampling: nearest
URI = 'netcdf:"/Users/halverso/data/GEOS5FP/2021.05.29/GEOS.fp.asm.tavg3_2d_aer_Nx.20210529_1930.V01.nc4":TOTEXTTAU'
nodata = np.nan
geometry = Point(-110.0522, 31.7438)
resampling = 'nearest'
raster = Raster.open(URI, nodata=nodata, geometry=geometry, resampling=resampling)
print(raster)
