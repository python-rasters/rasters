# `rasters` Repository History (`v1.0.2` – `v1.21.0`)

**Repository:** `python-rasters/rasters`  
**Author:** Gregory H. Halverson (NASA Jet Propulsion Laboratory)  
**Span:** June 13, 2024 (`v1.0.2`) – November 17, 2026 (`v1.21.0`)  
**Total Releases:** 53

---

## 1. Executive Summary

Across 53 releases over two and a half years, `rasters` evolved from a single 4,817-line monolithic module (`rasters/rasters.py`) adapted from ECOSTRESS workflows at NASA JPL into a modular, 29-module Python geospatial processing toolkit supporting Python 3.11 through 3.14.

The repository's evolution spans four distinct architectural eras:

1. **Monolithic Genesis (`v1.0.2` – `v1.2.0`):** Single-file architecture, pivot from Shapely multiple inheritance to composition, and `msgpack`-backed `KDTree` serialization.
2. **Modularization & Dependency Cleanup (`v1.3.0` – `v1.4.6`):** Decomposition into single-responsibility modules, removal of `pygeos`, point extraction, grid shifting, and macOS `pykdtree` compatibility.
3. **Analytical & Spatial Expansion (`v1.5.0` – `v1.16.1`):** Image downscaling/bias-correction, multi-raster mosaicking, swath subsetting, square-bracket geometry indexing, and `GeoSeries` integration.
4. **Stabilization & Modernization (`v1.17.0` – `v1.21.0`):** Full API docstrings and README documentation, Python 3.11–3.14 CI matrices, and elimination of Shapely 2.2+, `affine` 2.4+, and NumPy 2.0+ deprecation warnings.

---

## 2. Architectural Eras

### Era 1: Monolithic Genesis (`v1.0.2` – `v1.2.0`, Jun – Sep 2024)

* **`v1.0.2` – `v1.0.4` (Jun 13–20, 2024):**
  * Initial commit introduced `rasters/rasters.py` (4,817 lines) containing `CRS`, `SpatialGeometry`, `CoordinateArray`, `VectorGeometry`, `BBox`, `KDTree`, `RasterGeometry`, `RasterGeolocation`, `RasterGrid`, `Raster`, and `MultiRaster`.
  * Packaged for PyPI (`pip install rasters`) and promoted `pykdtree` from an optional macOS dependency to a core dependency.
* **`v1.0.5` – `v1.0.6` (Jun 20, 2024):**
  * Refactored vector classes (`Point`, `MultiPoint`, `Polygon`, `MultiPolygon`) from subclassing `shapely.geometry.*` directly to wrapping `self.geometry` via composition so CRS attributes persist across geometric operations.
  * Switched default colormap constant (`DEFAULT_CMAP`) from `"viridis"` to `"jet"`.
* **`v1.1.0` (Jun 21, 2024):**
  * Added `KDTree.to_dict()`, `KDTree.from_dict()`, `KDTree.save()`, and `KDTree.load()` using `msgpack` and `msgpack_numpy` to cache precomputed neighbor lookup arrays on disk.
* **`v1.1.1` – `v1.1.5` (Jul 19 – Aug 16, 2024):**
  * Fixed binary integer raster rendering (`[0, 1]`) and `MultiRaster` plotting/reprojection.
  * Added `.wkt` property and WKT `__repr__` output to `VectorGeometry`.
  * Updated `RasterGeolocation.index_point()` to use `scipy.spatial.cKDTree`.
* **`v1.2.0` (Sep 19, 2024):**
  * Restored `CoordinateArray` (temporarily removed in `v1.0.5`) to preserve compatibility with downstream ECOSTRESS pipelines, and configured `setuptools.build_meta` in `pyproject.toml`.

---

### Era 2: Modularization & Dependency Cleanup (`v1.3.0` – `v1.4.6`, Sep 2024 – Feb 2025)

* **`v1.3.0` (Sep 28, 2024):**
  * **Major Architectural Refactor:** Split the 4,800-line `rasters/rasters.py` monolith into dedicated single-responsibility modules (`CRS.py`, `bbox.py`, `constants.py`, `coordinate_array.py`, `spatial_geometry.py`, `vector_geometry.py`, `point.py`, `multi_point.py`, `polygon.py`, `multi_polygon.py`, `wrap_geometry.py`, `kdtree.py`, `raster_geometry.py`, `raster_geolocation.py`, `raster_grid.py`, `raster.py`, `multi_raster.py`, `out_of_bounds_error.py`, `where.py`, `clip.py`, `mosaic.py`).
  * Removed the deprecated `pygeos` dependency, migrating `RasterGeometry.pixel_outlines` to native `shapely.multipolygons()`.
  * Replaced deprecated `scipy.ndimage.interpolation.shift` imports with `scipy.ndimage.shift`.
* **`v1.3.1` – `v1.3.6` (Oct 29 – Nov 9, 2024):**
  * Resolved circular imports across newly split modules, extracted `center_aeqd.py` and `local_UTM_proj4.py`, and fixed a property recursion bug in `CRS._crs`.
  * Added `Raster.open(..., geometry=Point)` and `Raster.to_point()` with `"nearest"` and Inverse Distance Weighting (`IDW`) sampling modes.
  * Fixed `BBox.merge()` `ymax` calculation (`max` instead of `min`) and added `MultiVectorGeometry.__getitem__`.
* **`v1.4.0` – `v1.4.6` (Nov 22, 2024 – Feb 26, 2025):**
  * Added `RasterGrid.shift_xy()` and `RasterGrid.shift_distance()` for sub-pixel and metric grid translations.
  * Added `MultiRaster.contain()` to preserve 3D multi-band structure during reprojection and ufunc operations.
  * Pinned `pykdtree>=1.3.11` to resolve macOS wheel compilation issues and standardized `requires-python = ">=3.10"`.

---

### Era 3: Analytical & Spatial Expansion (`v1.5.0` – `v1.16.1`, Apr – Nov 2025)

* **`v1.5.0` – `v1.6.0` (Apr 8 – May 28, 2025):**
  * Added `linear_downscale()` and `bias_correct()` for coarse-to-fine thermal/optical image fusion (`v1.5.0`).
  * Fixed `RasterGeometry.__repr__` JSON serialization for NumPy scalar types (`v1.5.1`).
  * Fixed multi-band GeoTIFF writing in `Raster.to_file()` (`v1.5.3`) and added `MultiRaster.open()` (`multi = True`) plus configurable `resampling` options in `mosaic()` (`v1.5.4` – `v1.6.0`).
  * Added `BBox.centroid`, keyword `Point(x=..., y=...)` initialization, and initial `MultiPoint` coordinate properties (`v1.6.0`).
* **`v1.7.0` – `v1.11.0` (Jul 3 – Aug 19, 2025):**
  * Expanded `MultiPoint` to support `x`/`y` array initialization, 3D `z` coordinates, `to_crs()`, `latlon`, `lat`, and `lon`, formally deprecating `CoordinateArray` (`v1.7.0`).
  * Added `MultiPoint` extraction in `Raster.open()` (`v1.7.0`) and `RasterGeometry` pass-through in `wrap_geometry()` (`v1.7.1`).
  * Added `RasterGeolocation.window()` for swath spatial windowing (`v1.8.0`).
  * Updated `Raster.__array_wrap__()` for NumPy 2.0 compatibility and migrated `__version__` to `importlib.metadata.version("rasters")` (`v1.9.0`).
  * Fixed UTM zone clamping and suppressed pyproj PROJ4 conversion warnings in `local_UTM_proj4()` (`v1.10.0`).
  * Standardized `transform_xy()` with geographic bounds clipping and added multi-band support to `KDTree.resample()` (`v1.11.0`).
* **`v1.12.0` – `v1.16.1` (Sep 11 – Nov 7, 2025):**
  * Added `GeoSeries` of `Point` conversion and `None` pass-through in `wrap_geometry()` (`v1.12.0` – `v1.12.3`).
  * Added `Point.lat` and `Point.lon` properties (`v1.13.0` – `v1.13.1`).
  * Defaulted empty/blank CRS strings to `WGS84` in `CRS.__init__()` to fix NetCDF subdataset loading (`v1.13.2`).
  * Added `xmin`, `xmax`, `ymin`, and `ymax` convenience properties to `RasterGrid` (`v1.14.0`).
  * Added `RasterGeolocation.subset()`, `window(..., buffer=...)`, and square-bracket geometry indexing (`geometry[polygon]`, `geometry[window]`) on `RasterGeometry` (`v1.15.0`).
  * Fixed 3D `MultiRaster` output in `where()` and `mosaic()` (`v1.15.1` – `v1.16.1`).

---

### Era 4: Stabilization & Modernization (`v1.17.0` – `v1.21.0`, Jan – Nov 2026)

* **`v1.17.0` – `v1.17.1` (Jan 13, 2026):**
  * Updated `RasterGrid.from_bbox()` to accept `xmin`, `ymin`, `xmax`, `ymax` keyword arguments directly.
  * Added comprehensive Google-style docstrings across `Raster`, `RasterGrid`, `RasterGeolocation`, and updated `README.md`.
* **`v1.18.0` – `v1.20.0` (Mar 5 – Jul 31, 2026):**
  * Added `BBox.boundary_latlon` (`v1.18.0`) and expanded CI testing across Python 3.11–3.14.
  * Fixed `BBox` handling in `wrap_geometry()` (`v1.19.0`).
  * Added `geopandas.array.GeometryArray` clipping support to `clip()` and updated PyPI release workflows (`v1.20.0`).
* **`v1.21.0` (Nov 17, 2026):**
  * Resolved remaining deprecation warnings under modern dependencies (GitHub issues `#69` and `#47`):
    * Migrated `VectorGeometry.to_crs()` and `RasterGeometry.crosses_antimeridian` from `shapely.ops.transform()` to vectorized `shapely.transform()`.
    * Replaced deprecated `affine.Affine` `*` multiplication with `@` matrix multiplication across `RasterGrid`.
    * Added regression tests in `tests/test_vector_geometry.py` enforcing zero `DeprecationWarning` or `PendingDeprecationWarning` emissions.

---

## 3. Complete Release Index

| Tag | Release Date | Summary |
| :--- | :--- | :--- |
| `v1.21.0` | 2026-11-17 | Resolve Shapely, `affine`, and NumPy deprecation warnings (`#69`, `#47`) |
| `v1.20.0` | 2026-07-31 | Fix PyPI authentication, Python 3.11–3.14 support, `GeometryArray` support in `clip()` |
| `v1.19.0` | 2026-07-27 | Fix `BBox` pass-through handling in `wrap_geometry()` |
| `v1.18.0` | 2026-03-05 | Add `BBox.boundary_latlon`; CI matrix Python 3.10–3.14 |
| `v1.17.1` | 2026-01-13 | Fix top-level geometry imports in `raster_grid.py` |
| `v1.17.0` | 2026-01-13 | Add docstrings & `README.md` guide; support `xmin`/`ymin`/`xmax`/`ymax` in `RasterGrid.from_bbox()` |
| `v1.16.1` | 2025-11-07 | Fix shape check typo in `mosaic()` for `MultiRaster` inputs |
| `v1.16.0` | 2025-11-07 | Add `MultiRaster` support and optional `geometry` parameter to `mosaic()` |
| `v1.15.1` | 2025-11-06 | Fix `MultiRaster` masking by returning `MultiRaster` from `where()` for 3D arrays |
| `v1.15.0` | 2025-11-06 | Add `RasterGeolocation.subset()` and square-bracket geometry indexing on `RasterGeometry` |
| `v1.14.0` | 2025-10-24 | Add `xmin`, `xmax`, `ymin`, `ymax` bounds attributes to `RasterGrid` |
| `v1.13.2` | 2025-10-24 | Default empty/None CRS parameters to `WGS84` for NetCDF subdatasets |
| `v1.13.1` | 2025-10-24 | Optimize `Point.lat` and `Point.lon` when already in a geographic CRS |
| `v1.13.0` | 2025-10-24 | Add `lat` and `lon` properties to `Point` |
| `v1.12.3` | 2025-09-11 | Return `None` when `wrap_geometry(None)` is called |
| `v1.12.2` | 2025-09-11 | Fix missing `GeoSeries` import in `wrap_geometry.py` |
| `v1.12.0` | 2025-09-11 | Support `GeoSeries` of `Point` to `MultiPoint` conversion in `wrap_geometry()` |
| `v1.11.0` | 2025-08-19 | Add `transform_xy()` module and multi-band resampling support in `KDTree.resample()` |
| `v1.10.0` | 2025-08-19 | Fix UTM zone clamping and suppress PROJ4 conversion warnings in `local_UTM_proj4()` |
| `v1.9.0` | 2025-08-06 | Update `Raster.__array_wrap__` for NumPy 2.0; switch version metadata to `importlib.metadata` |
| `v1.8.0` | 2025-07-24 | Add `RasterGeolocation.window()` for swath spatial subsetting |
| `v1.7.1` | 2025-07-22 | Fix `wrap_geometry()` pass-through behavior for `RasterGeometry` inputs |
| `v1.7.0` | 2025-07-03 | Expand `MultiPoint` (`x`/`y` arrays, `to_crs`, `lat`/`lon`), deprecate `CoordinateArray`, support `MultiPoint` in `Raster.open()` |
| `v1.6.0` | 2025-05-28 | Add `resampling` parameter to `mosaic()`; add `BBox.centroid` and `MultiPoint` coordinate properties |
| `v1.5.5` | 2025-05-28 | Suppress runtime NaN warnings during `mosaic()` compositing |
| `v1.5.4` | 2025-05-28 | Support multi-band `MultiRaster.open()` and export `mosaic` in package namespace |
| `v1.5.3` | 2025-05-20 | Fix multi-band GeoTIFF writing in `Raster.to_file()` |
| `v1.5.2` | 2025-04-24 | Fix `KDTree` import in `raster.py` |
| `v1.5.1` | 2025-04-22 | Cast NumPy scalars to native Python `int`/`float` in `RasterGeometry.__repr__` JSON output |
| `v1.5.0` | 2025-04-08 | Add `linear_downscale()` and `bias_correct()` functions |
| `v1.4.6` | 2025-02-26 | Finalize `pyproject.toml` metadata for Python `>=3.10` |
| `v1.4.5` | 2025-02-26 | Set minimum Python version to `>=3.10` |
| `v1.4.4` | 2025-02-03 | Require `pykdtree>=1.3.11` to resolve macOS build issues |
| `v1.4.3` | 2025-01-03 | Add `MultiRaster.contain()` to fix `MultiRaster` reprojection |
| `v1.4.2` | 2024-11-22 | Fix `RasterGrid.shift_xy()` scaling by dividing shift by `cell_size` |
| `v1.4.1` | 2024-11-22 | Intermediate fix for `RasterGrid.shift_xy()` origin calculation |
| `v1.4.0` | 2024-11-22 | Add `RasterGrid.shift_xy()` and `RasterGrid.shift_distance()` |
| `v1.3.6` | 2024-11-09 | Fix `Raster` import inside `where()` |
| `v1.3.5` | 2024-11-06 | Return Python scalar via `.item()` in `Raster.to_point()` |
| `v1.3.4` | 2024-11-06 | Fix recursion bug in `CRS._crs`; add initial `Raster.to_point()` extraction |
| `v1.3.3` | 2024-10-31 | Add module docstrings, `center_aeqd()`, `local_UTM_proj4()`, and fix `BBox.merge()` `ymax` |
| `v1.3.2` | 2024-10-30 | Fix circular `RasterGeolocation` and `CRS` imports in `raster.py` |
| `v1.3.1` | 2024-10-29 | Fix `RasterGeometry` and `RasterGrid` imports and boundary index generation |
| `v1.3.0` | 2024-09-28 | Remove `pygeos` dependency and split monolithic `rasters.py` into modular files |
| `v1.2.0` | 2024-09-19 | Restore `CoordinateArray` class and configure `setuptools.build_meta` |
| `v1.1.5` | 2024-08-16 | Switch `RasterGeolocation.index_point()` to `scipy.spatial.cKDTree` |
| `v1.1.4` | 2024-08-16 | Add `.wkt` property and WKT `__repr__` to `VectorGeometry` |
| `v1.1.3` | 2024-08-15 | Fix destination array allocation for 3D multi-band rasters in `Raster.to_grid()` |
| `v1.1.2` | 2024-08-15 | Fix `plt.xticks(rotation=-90)` argument type in `MultiRaster.imshow()` |
| `v1.1.1` | 2024-07-19 | Fix binary `[0, 1]` integer raster colormap detection in `Raster.imshow()` |
| `v1.1.0` | 2024-06-21 | Add `KDTree.to_dict()`, `from_dict()`, `save()`, and `load()` via `msgpack` |
| `v1.0.6` | 2024-06-20 | Remove stray `import raster` from `rasters.py` |
| `v1.0.5` | 2024-06-20 | Refactor vector geometries (`Point`, `MultiPoint`, `Polygon`, `MultiPolygon`) to composition |
| `v1.0.4` | 2024-06-20 | Add PyPI packaging metadata and `pykdtree` dependency |
| `v1.0.2` | 2024-06-20 | Initial release of `rasters` package |