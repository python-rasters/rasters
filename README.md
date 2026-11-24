# `rasters` python package

![CI](https://github.com/python-rasters/rasters/actions/workflows/ci.yml/badge.svg)

This Python software package provides a comprehensive solution for handling gridded and swath raster data. It offers an object-oriented interface for ease of use, treating rasters, raster geometries, vector geometries, and K-D trees as objects. This allows for seamless map visualization in Jupyter notebooks and efficient resampling between swath and grid geometries.

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 321H

## Citing this Package

If you use `rasters` in your research or software pipelines, please cite it using the following BibTeX entry:

```bibtex
@software{halverson_rasters_2026,
  author       = {Halverson, Gregory H.},
  title        = {rasters: Object-Oriented Gridded and Swath Raster Processing Toolkit for Python},
  year         = {2026},
  version      = {1.21.0},
  publisher    = {GitHub},
  url          = {https://github.com/python-rasters/rasters}
}
```

The software addresses several challenges in the field. It makes swath remote sensing datasets more accessible and easier to work with. It also associates coordinate reference systems with vector geometries, simplifying transformations between projections.

The software improves upon existing packages such as rasterio, shapely, pyresample, and GDAL by encapsulating functionalities into objects and providing a more user-friendly interface. It is inspired by and aims to be a Python equivalent for the raster package in R and the Rasters.jl package in Julia.

Developed under a research grant by the NASA Research Opportunities in Space and Earth Sciences (ROSES) program at the Jet Propulsion Laboratory (JPL), the software was designed for use by the Ecosystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) mission and the Surface Biology and Geology (SBG) mission. However, its utility extends to general remote sensing and GIS projects in Python.

The software has potential commercial applications in remote sensing data analysis and pipeline construction. It meets the SPD-41 open-science requirements of NASA-funded ROSES projects by being published as an open-source software package.

The advantage of this software package is that it brings together common geospatial operations in a single easy to use interface. It is useful for both remote sensing data analysis and building remote sensing data pipelines. It is anticipated to be of interest to those involved in remote sensing and GIS projects in Python.

### This software accomplishes the following:

This python package handles reading, writing, visualizing, and resampling of gridded and swath raster data.

### What are the unique features of the software?
- object oriented interface for ease of use
- rasters and raster geometries as objects
- map visualization of raster objects in Jupyter notebooks
- ability to sample back and forth between swath and grid geometries
- vector geometries with associated coordinate reference system

### What improvements have been made over existing similar software application?

This software improves over the rasterio package by encapsulating rasters as objects that generate map visualizations when inspected in a Jupyter notebook. This software improves over the shapely package by associating coordinate reference systems with vector geometries, making it easier to transform between projections. This software improves over the pyresample package by encapsulating swath and grid geometries as objects and making it easy to resample between them. This software improves over GDAL with an object-oriented interface that can read and write a variety of raster file formats.

### What problems are you trying to solve in the software?

This software solves the problem of swath remote sensing datasets being inaccessible and difficult to work with. This software solves the problem of vector geometries in python not having coordinate reference systems associated with them. The software is inspired by and intended to be a python equivalent for the raster package in R and the Rasters.jl package in Julia.

### Does your work relate to current or future NASA (include reimbursable) work that has value to the conduct of aeronautical and space activities?  If so, please explain:

This software package was developed as part of a research grant by the NASA Research Opportunities in Space and Earth Sciences (ROSES) program. This software was designed for use by the Ecosystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) mission as a precursor for the Surface Biology and Geology (SBG) mission, but it may be useful generally for remote sensing and GIS projects in python.

### What advantages does this software have over existing software?

The advantage of this software package is that it brings together common geospatial operations in a single easy to use interface.

Are there any known commercial applications? What are they? What else is currently on the market that is similar?

This software is useful for both remote sensing data analysis and building remote sensing data pipelines.

### Is anyone interested in the software? Who? Please list organization names and contact information.

- NASA ROSES
- ECOSTRESS
- SBG

### What are the current hardware and operating system requirements to run the software? (Platform, RAM requirement, special equipment, etc.) 

This software is written entirely in python and intended to be distributed using the pip package manager.

### How has the software performed in tests? Describe further testing if planned. 

This software has been deployed for ECOSTRESS and ET-Toolbox.

### Please identify the customer(s) and sponsors(s) outside of your section that requested and are using your software. 

This package is being released according to the SPD-41 open-science requirements of NASA-funded ROSES projects.

## Installation

The `rasters` package is available as a [pip package on PyPi](https://pypi.org/project/rasters/):

```
pip install rasters
```

## Examples

Import the `Raster` class from the `rasters` package.

```python
from rasters import Raster
```

Supply the filename to the `open` class method of the `Raster` class. Placing the variable for the `Raster` object at the end of a Jupyter notebook cell displays a map of the image. The default `cmap` used in the map is `viridis`.

```python
raster = Raster.open("ECOv002_L2T_LSTE_33730_008_11SPS_20240617T205018_0712_01_LST.tif")
raster
```

![png](examples/Opening%20a%20GeoTIFF_3_0.png)

## Usage

### Working with Raster Objects

The `Raster` class is the primary interface for working with raster data. It encapsulates both the data array and its georeferencing information.

#### Opening and Creating Rasters

```python
from rasters import Raster

# Open a raster from file
raster = Raster.open("path/to/file.tif")

# Create a raster from a numpy array and geometry
import numpy as np
from rasters import RasterGrid

data = np.random.rand(100, 100)
geometry = RasterGrid.from_bbox(
    xmin=-120, ymin=30, xmax=-110, ymax=40,
    cell_size=0.1
)
raster = Raster(data, geometry=geometry)
```

#### Raster Properties

```python
# Access the data array
array = raster.array

# Get dimensions
rows, cols = raster.shape
print(f"Raster size: {rows} rows x {cols} cols")

# Get coordinate reference system
crs = raster.crs

# Get bounding box
bbox = raster.bbox

# Get cell size
cell_size = raster.cell_size
```

#### Raster Operations

```python
# Arithmetic operations
raster_sum = raster1 + raster2
raster_diff = raster1 - raster2
raster_product = raster1 * raster2
raster_quotient = raster1 / raster2

# Comparison operations
mask = raster > 0.5

# Reproject to different coordinate system
reprojected = raster.reproject(crs="EPSG:4326", target_cell_size=0.01)

# Resample to a different geometry
resampled = raster.to_geometry(target_geometry, resampling="bilinear")

# Sample at a point
from rasters import Point
point = Point(-115, 35, crs="EPSG:4326")
value = raster.to_point(point)
```

#### Saving Rasters

```python
# Save as GeoTIFF
raster.to_geotiff("output.tif", compression="deflate")

# Save as Cloud Optimized GeoTIFF (COG)
raster.to_COG("output_cog.tif")

# Save as GeoPackage
raster.to_geopackage("output.gpkg")
```

#### Visualization

```python
# Display in Jupyter notebook (automatic when last line in cell)
raster

# Create a matplotlib figure
fig = raster.imshow(
    title="My Raster",
    cmap="viridis",
    figsize=(10, 8)
)

# Save as image
image = raster.to_pillow(cmap="jet")
image.save("preview.png")
```

### Working with RasterGrid

The `RasterGrid` class represents gridded raster geometry with uniform cell spacing and north-oriented grids.

#### Creating RasterGrid Objects

```python
from rasters import RasterGrid

# From bounding box and cell size
grid = RasterGrid.from_bbox(
    xmin=-120, ymin=30, xmax=-110, ymax=40,
    cell_size=0.001,
    crs="EPSG:4326"
)

# From bounding box and shape
grid = RasterGrid.from_bbox(
    xmin=-120, ymin=30, xmax=-110, ymax=40,
    shape=(1000, 1000),
    crs="EPSG:4326"
)

# From affine transform
from affine import Affine
affine = Affine(0.001, 0, -120, 0, -0.001, 40)
grid = RasterGrid.from_affine(affine, rows=1000, cols=1000, crs="EPSG:4326")

# From coordinate vectors
x_vector = np.linspace(-120, -110, 1000)
y_vector = np.linspace(30, 40, 1000)
grid = RasterGrid.from_vectors(x_vector, y_vector, crs="EPSG:4326")

# From a raster file
grid = RasterGrid.open("path/to/file.tif")
```

#### RasterGrid Properties

```python
# Get grid dimensions
rows = grid.rows
cols = grid.cols

# Get cell size
cell_width = grid.cell_width
cell_height = grid.cell_height

# Get extent
xmin, ymin = grid.xmin, grid.ymin
xmax, ymax = grid.xmax, grid.ymax
width, height = grid.width, grid.height

# Get coordinate arrays
x = grid.x  # 2D array of x-coordinates
y = grid.y  # 2D array of y-coordinates

# Get coordinate vectors
x_vector = grid.x_vector  # 1D array of x-coordinates
y_vector = grid.y_vector  # 1D array of y-coordinates

# Get affine transform
affine = grid.affine
```

#### RasterGrid Operations

```python
# Subset to a region
from rasters import BBox
subset_bbox = BBox(-115, 32, -112, 38, crs="EPSG:4326")
subset_grid = grid.subset(subset_bbox)

# Slice using indices
subset_grid = grid[100:200, 150:250]

# Change resolution
rescaled_grid = grid.rescale(cell_size=0.002)

# Buffer the grid
buffered_grid = grid.buffer(pixels=10)

# Shift the grid
shifted_grid = grid.shift_xy(x_shift=100, y_shift=50)
```

### Working with RasterGeolocation

The `RasterGeolocation` class represents swath raster geometry using 2D geolocation arrays, commonly used for satellite swath data.

#### Creating RasterGeolocation Objects

```python
from rasters import RasterGeolocation
import numpy as np

# From x and y coordinate arrays
x_coords = np.random.uniform(-120, -110, (500, 500))
y_coords = np.random.uniform(30, 40, (500, 500))
geolocation = RasterGeolocation(x_coords, y_coords, crs="EPSG:4326")

# From coordinate vectors (creates meshgrid)
x_vector = np.linspace(-120, -110, 500)
y_vector = np.linspace(30, 40, 500)
geolocation = RasterGeolocation.from_vectors(x_vector, y_vector, crs="EPSG:4326")
```

#### RasterGeolocation Properties

```python
# Get dimensions
rows = geolocation.rows
cols = geolocation.cols

# Get coordinate arrays
x = geolocation.x  # 2D array of x-coordinates
y = geolocation.y  # 2D array of y-coordinates

# Get extent
xmin, ymin = geolocation.x_min, geolocation.y_min
xmax, ymax = geolocation.x_max, geolocation.y_max

# Get cell size (estimated from median distances)
cell_size = geolocation.cell_size

# Get boundary polygon
boundary = geolocation.boundary
```

#### RasterGeolocation Operations

```python
# Index with geometry
from rasters import Polygon
polygon = Polygon([(-115, 32), (-112, 32), (-112, 38), (-115, 38)])
mask = geolocation.index(polygon)

# Generate a regular grid from geolocation
grid = geolocation.grid

# Resize geolocation arrays
resized = geolocation.resize(dimensions=(250, 250))

# Subset to a region
subset = geolocation.subset(polygon)
```

### Working with Points and Vector Geometries

```python
from rasters import Point, Polygon, BBox

# Create a point
point = Point(-115.5, 35.2, crs="EPSG:4326")

# Transform point to different CRS
utm_point = point.to_crs("EPSG:32611")

# Create a polygon
polygon = Polygon([
    (-115, 32),
    (-112, 32),
    (-112, 38),
    (-115, 38)
], crs="EPSG:4326")

# Create a bounding box
bbox = BBox(xmin=-120, ymin=30, xmax=-110, ymax=40, crs="EPSG:4326")

# Sample raster at point
value = raster.to_point(point)

# Clip raster to polygon
clipped = raster.subset(polygon)
```

### Advanced Usage

#### Merging Multiple Rasters

```python
# Open multiple rasters
raster1 = Raster.open("tile1.tif")
raster2 = Raster.open("tile2.tif")
raster3 = Raster.open("tile3.tif")

# Merge into a single raster
merged = Raster.merge([raster1, raster2, raster3])
```

#### Resampling with KDTree

```python
from rasters import KDTree

# Create a KDTree for efficient resampling
kd_tree = KDTree(
    source_geometry=source_raster.geometry,
    target_geometry=target_grid,
    radius_of_influence=1000  # meters
)

# Resample using the KDTree
resampled = source_raster.resample(target_grid, kd_tree=kd_tree)

# Save KDTree for reuse
kd_tree.save("kdtree.pkl")

# Load KDTree
kd_tree = KDTree.load("kdtree.pkl")
```

#### Working with Masks

```python
# Create a mask
mask = raster > 0.5

# Apply mask to raster
masked_raster = raster.mask(mask)

# Fill masked values with another raster
filled = raster.fill(fill_raster)
```

## History

* **`v1.0.2` – `v1.2.0` (June – September 2024):** Launched as a single-file package (`rasters/rasters.py`) adapted from ECOSTRESS workflows at NASA JPL. Refactored vector geometry classes (`Point`, `MultiPoint`, `Polygon`, `MultiPolygon`) from Shapely inheritance to composition with explicit CRS tracking, added `msgpack` serialization (`KDTree.save()` and `KDTree.load()`), added `.wkt` representations, and switched swath point lookup to `scipy.spatial.cKDTree`.
* **`v1.3.0` – `v1.4.6` (September 2024 – February 2025):** Split the monolith into dedicated single-responsibility modules (`CRS.py`, `bbox.py`, `point.py`, `polygon.py`, `kdtree.py`, `raster.py`, `raster_grid.py`, `raster_geolocation.py`, etc.), dropped the deprecated `pygeos` dependency in favor of `shapely` 2.x, added point extraction via `Raster.open(..., geometry=Point)`, added grid shifting (`RasterGrid.shift_xy()`, `shift_distance()`), and resolved macOS `pykdtree` build issues (`pykdtree>=1.3.11`).
* **`v1.5.0` – `v1.16.1` (April – November 2025):** Added `linear_downscale()` and `bias_correct()`, configurable resampling in `mosaic()`, `MultiPoint` extraction in `Raster.open()`, `RasterGeolocation.window()` and `.subset()`, square-bracket geometry indexing (`grid[polygon]`, `geolocation[window]`), `GeoSeries` to `MultiPoint` wrapping, `Point.lat`/`Point.lon` properties, `RasterGrid` bounds properties (`xmin`, `xmax`, `ymin`, `ymax`), and multi-raster mosaicking/masking fixes.
* **`v1.17.0` – `v1.21.0` (January – November 2026):** Expanded module docstrings and documentation, added `BBox.boundary_latlon`, expanded CI test coverage across Python `3.11`–`3.14`, fixed `BBox` handling in `wrap_geometry()`, and resolved deprecation warnings under Shapely 2.2+ (`shapely.transform()`), `affine` 2.4+ (`@` matrix multiplication), and NumPy 2.0+ (`__array_wrap__`).

## Known Issues

The following known issues are tracked in the [GitHub issue tracker](https://github.com/python-rasters/rasters/issues) as of `v1.21.0`:

* **Vector & Geometry Edge Cases:**
  * Calling `Polygon()` or `MultiPolygon()` with no arguments raises an `IndexError` instead of initializing an empty geometry ([#74](https://github.com/python-rasters/rasters/issues/74)).
  * `RasterGeometry.point` returns a `ValueError` instance rather than raising it when the geometry is not a single cell ([#78](https://github.com/python-rasters/rasters/issues/78)).
  * `BBox.round` truncates coordinates to whole integers without supporting an `ndigits` precision argument ([#85](https://github.com/python-rasters/rasters/issues/85)).
  * `rasters/local_UTM_proj4_from_lat_lon.py` is an orphaned stub file with an empty function definition superseded by `rasters/local_UTM_proj4.py` ([#71](https://github.com/python-rasters/rasters/issues/71)).
* **Raster Indexing, Slicing & Grid Operations:**
  * `Raster.__setitem__()` fails on slice and tuple-slice assignments (`raster[0:5] = 0`, `raster[0:5, 0:5] = 0`) ([#75](https://github.com/python-rasters/rasters/issues/75)).
  * Single-dimension indexing (`raster[0]` or `raster[0:5]`) raises a `TypeError` in `Raster._key()` ([#76](https://github.com/python-rasters/rasters/issues/76)).
  * `Raster._slice()` mutates array view shapes in-place (`result.shape = new_shape`), failing on non-contiguous slices ([#82](https://github.com/python-rasters/rasters/issues/82)).
  * `RasterGrid.resolution()` raises a `TypeError` when passed a scalar `float` instead of a tuple ([#77](https://github.com/python-rasters/rasters/issues/77)).
  * `Raster.origin` delegates to a nonexistent `self.geometry.origin` attribute ([#79](https://github.com/python-rasters/rasters/issues/79)).
  * `Raster.rescale()` resizes the data array to the original `(self.rows, self.cols)` shape instead of the target `(geometry.rows, geometry.cols)` shape ([#83](https://github.com/python-rasters/rasters/issues/83)).
* **MultiRaster, Mosaicking & Reprojection:**
  * `MultiRaster.contain()` / `MultiRaster.__init__()` nests `metadata` under a `"metadata"` key instead of unpacking it ([#80](https://github.com/python-rasters/rasters/issues/80)).
  * `Raster.to_grid()` raises a `TypeError` on boolean rasters when `nodata` is `None` ([#81](https://github.com/python-rasters/rasters/issues/81)).
  * `mosaic()` fails on generator/iterator inputs and evaluates `geometry.shape` before resolving default geometry from `images[0]` ([#84](https://github.com/python-rasters/rasters/issues/84)).
* **File I/O, Testing & Warnings:**
  * `Raster.from_geotiff()` calls a nonexistent `cls.from_file()` method and rejects `.tiff` extensions due to operator precedence ([#72](https://github.com/python-rasters/rasters/issues/72)).
  * `Raster.to_COG()` invokes `gdal_translate` via `os.system()` without checking exit status and uses naive `.replace(".tif", ".temp.tif")` path substitution ([#87](https://github.com/python-rasters/rasters/issues/87)).
  * `RasterGeometry` uses `warnings.filterwarnings('ignore')` followed by `warnings.resetwarnings()` instead of `with warnings.catch_warnings()`, clearing external warning filters ([#86](https://github.com/python-rasters/rasters/issues/86)).
  * `tests/test_write_geotiff.py` is currently an empty placeholder without GeoTIFF write tests ([#73](https://github.com/python-rasters/rasters/issues/73)).