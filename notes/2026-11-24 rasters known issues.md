# `rasters` Known Issues & Codebase Audit (`v1.21.0`)

This document catalogs bugs, dead code, edge-case runtime exceptions, and subtle logic flaws identified across the `rasters` repository (`v1.21.0`).

---

## 1. Broken & Dead Code

### 1.1 `SyntaxError` in `rasters/local_UTM_proj4_from_lat_lon.py`
* **Location:** `rasters/local_UTM_proj4_from_lat_lon.py:1`
* **Issue:** The file contains an empty function signature with no body (`def local_UTM_proj4_from_lat_lon(lat: float, lon: float) -> str:`), raising a `SyntaxError` if imported.
* **Context:** Superseded by `rasters/local_UTM_proj4.py` in `v1.10.0`.
* **Fix:** Delete `rasters/local_UTM_proj4_from_lat_lon.py`.

### 1.2 Broken `Raster.from_geotiff()` Method
* **Location:** `rasters/raster.py` (`Raster.from_geotiff`)
* **Issues:**
  1. Calls `cls.from_file(filename, **kwargs)`, which does not exist on `Raster` (the method is `cls.open(...)`).
  2. Extension validation `if not filename.endswith(".tif") or filename.endswith(".tiff"):` has inverted operator precedence, causing valid `.tiff` filenames to raise `ValueError("invalid GeoTIFF filename")`.
* **Fix:** Check `not filename.lower().endswith((".tif", ".tiff"))` and delegate to `cls.open(filename, **kwargs)`.

### 1.3 Empty Test File `tests/test_write_geotiff.py`
* **Location:** `tests/test_write_geotiff.py`
* **Issue:** Exists as a 0-byte file in the test suite with no tests implemented.
* **Fix:** Add unit tests covering `Raster.to_geotiff()`, `MultiRaster.to_geotiff()`, `Raster.to_COG()`, and `Raster.to_geopackage()`.

---

## 2. Runtime Exceptions on Valid Edge-Case Inputs

### 2.1 `IndexError` on Empty `Polygon()` and `MultiPolygon()` Constructors
* **Location:** `rasters/polygon.py` (`Polygon.__init__`), `rasters/multi_polygon.py` (`MultiPolygon.__init__`)
* **Issue:** Both constructors index `args[0]` directly without checking `if len(args) > 0:`. Calling `Polygon()` or `MultiPolygon()` with no arguments raises an `IndexError` instead of creating an empty geometry (unlike `Point()`, which checks `len(args) > 0`).
* **Fix:** Guard `args[0]` checks with `if len(args) > 0 and isinstance(args[0], ...)`.

### 2.2 Slice & Tuple Assignment Bugs in `Raster.__setitem__()`
* **Location:** `rasters/raster.py` (`Raster.__setitem__`)
* **Issues:**
  1. `if isinstance(key, slice): self.array[slice] = value` indexes with the built-in `slice` class instead of the `key` variable, raising `IndexError: only integers, slices... are valid indices`.
  2. Tuple slice assignments (e.g., `raster[0:5, 0:5] = 0`) fall through to `self.array[np.array(key)] = value`, which fails when `key` contains `slice` objects.
* **Fix:** Assign via `self.array[key] = value` for `slice`, `int`, `tuple`, `Raster`, and `np.ndarray` keys.

### 2.3 Single-Dimension Indexing Failure in `Raster._key()`
* **Location:** `rasters/raster.py` (`Raster._key`)
* **Issue:** Calling `raster[0]` or `raster[0:5]` (passing an `int` or `slice` instead of a 2-tuple) invokes `len(key)` without first wrapping `key = (key,)` as `RasterGeometry._key()` does, raising `TypeError: object of type 'int'/'slice' has no len()`.
* **Fix:** Normalize scalar/slice keys with `if isinstance(key, (int, slice)): key = (key,)`.

### 2.4 `RasterGrid.resolution()` Fails on Scalar `float`
* **Location:** `rasters/raster_grid.py` (`RasterGrid.resolution`)
* **Issue:** Evaluates `if len(cell_size) == 1:`, which raises `TypeError: object of type 'float' has no len()` when passed a single float, contradicting its `Union[float, Tuple[float, float]]` type annotation.
* **Fix:** Check `isinstance(cell_size, (int, float, np.number))` before checking `len(cell_size)`.

### 2.5 `RasterGeometry.point` Returns Exception Instead of Raising
* **Location:** `rasters/raster_geometry.py` (`RasterGeometry.point`)
* **Issue:** Executes `return ValueError("not a single point")` rather than raising the exception.
* **Fix:** Replace `return` with `raise`.

### 2.6 Missing `origin` Attribute in `Raster.origin`
* **Location:** `rasters/raster.py` (`Raster.origin`)
* **Issue:** Delegates to `self.geometry.origin`, which is not defined on `RasterGeometry`, `RasterGrid`, or `RasterGeolocation` (only `x_origin` and `y_origin` exist).
* **Fix:** Either return `(self.x_origin, self.y_origin)` on `RasterGrid` or remove the broken property.

---

## 3. Subtle Logic & Data-Handling Bugs

### 3.1 `MultiRaster.contain()` Drops Top-Level `metadata` Keys
* **Location:** `rasters/multi_raster.py` (`MultiRaster.__init__`, `MultiRaster.contain`)
* **Issue:** `MultiRaster.contain()` passes `metadata=self.metadata` to `MultiRaster(...)`, and `MultiRaster.__init__` loops `for key, value in kwargs.items(): self._metadata[key] = value` without explicitly accepting `metadata: dict = None` like `Raster.__init__` does. This nests the entire metadata dictionary under `self._metadata["metadata"]`.
* **Fix:** Add `metadata: dict = None` handling to `MultiRaster.__init__` matching `Raster.__init__`.

### 3.2 `TypeError` When Reprojecting Boolean `Raster` with `nodata=None`
* **Location:** `rasters/raster.py` (`Raster.to_grid`)
* **Issue:** When `self.dtype == "bool"` and `nodata is None`, `src_nodata = np.uint16(self.nodata)` attempts to cast `None` to `np.uint16`, raising `TypeError`.
* **Fix:** Default `src_nodata = 0` when `self.nodata is None` for boolean arrays.

### 3.3 In-Place Shape Mutation on Non-Contiguous Slices in `Raster._slice()`
* **Location:** `rasters/raster.py` (`Raster._slice`)
* **Issue:** Mutates `result.shape = new_shape` in-place on the sliced array view, which can raise `AttributeError: Incompatible shape for in-place modification` on non-contiguous views.
* **Fix:** Use `result = result.reshape(new_shape)`.

### 3.4 `Raster.rescale()` Resizes Array to Old Dimensions
* **Location:** `rasters/raster.py` (`Raster.rescale`)
* **Issue:** Computes the target `geometry = self.geometry.reproject(self.crs, target_cell_size)` with updated dimensions, but calls `skimage.transform.resize(self.array, (self.rows, self.cols), ...)` using the *original* rows and columns instead of `(geometry.rows, geometry.cols)`.
* **Fix:** Pass `(geometry.rows, geometry.cols)` as the target output shape.

### 3.5 `mosaic()` Ordering & Iterator Handling Bugs
* **Location:** `rasters/mosaic.py` (`mosaic`)
* **Issues:**
  1. Accepts `Iterator[...]` in its type signature, but calls `len(images)` and `images[0]`, which fail on generators/iterators.
  2. Accesses `geometry.shape` before the `if geometry is None: geometry = images[0].geometry` fallback check, raising `AttributeError: 'NoneType' object has no attribute 'shape'` when `geometry` is omitted.
* **Fix:** Materialize `images = list(images)` up front and resolve `if geometry is None:` before initializing `mosaic`.

### 3.6 `BBox.round` Truncates Coordinates to Whole Integers
* **Location:** `rasters/bbox.py` (`BBox.round`)
* **Issue:** Calls `round(self.x_min)` with no `ndigits` parameter, collapsing geographic bounding boxes (e.g., `-118.5, 34.5`) to coarse integer degrees (`-118, 34`).
* **Fix:** Support an optional `decimals: int = 0` method or document integer rounding behavior explicitly.

### 3.7 Global Warning Filter Reset in `RasterGeometry`
* **Location:** `rasters/raster_geometry.py` (`latlon_matrices`, `grid_to_crs`)
* **Issue:** Calls `warnings.filterwarnings('ignore')` followed by `warnings.resetwarnings()` instead of `with warnings.catch_warnings():`, clearing user-configured warning filters (such as `pytest -W error`).
* **Fix:** Scope warning suppression inside `with warnings.catch_warnings():`.

### 3.8 Unchecked `os.system` and Path Replacement in `Raster.to_COG()`
* **Location:** `rasters/raster.py` (`Raster.to_COG`)
* **Issues:**
  1. Uses `filename.replace(".tif", ".temp.tif")`, which corrupts paths if `.tif` appears in a parent directory name or if the file ends in `.tiff`.
  2. Executes `os.system(f'gdal_translate ...')` without checking the exit status or safely quoting arguments.
* **Fix:** Use `splitext(filename)[0] + ".temp.tif"` and `subprocess.run(..., check=True)`.