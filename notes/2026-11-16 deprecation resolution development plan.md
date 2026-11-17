# `rasters` Development Plan: Resolve Shapely, `affine`, and NumPy Deprecation Warnings

**Date:** 2026-11-16  
**Target Repository:** `rasters` (`1.20.0` $\rightarrow$ `1.21.0`)  
**Related GitHub Issues:** `#69`, `#47`  
**Affected Files:**
- `rasters/vector_geometry.py`
- `rasters/raster_geometry.py`
- `rasters/raster_grid.py`
- `tests/test_vector_geometry.py`
- `pyproject.toml`

---

## 1. Problem Summary

Running `rasters` under modern Shapely (2.2+), `affine` (2.4+), and NumPy (2.0+) emits deprecation warnings across core geometry and raster operations:

1. **Shapely `shapely.ops.transform()` (`#69`):**
   Calling `VectorGeometry.to_crs()` or evaluating `RasterGeometry.crosses_antimeridian` emits:
   ```text
   DeprecationWarning: The 'shapely.ops.transform()' function is deprecated. Use 'shapely.transform()' or 'shapely.transform_coordseq()' instead. See [https://shapely.readthedocs.io/en/latest/manual.html#shapely.ops.transform](https://shapely.readthedocs.io/en/latest/manual.html#shapely.ops.transform) for more details.
   ```
   Migrating to `shapely.transform()` eliminates the warning and passes `(N, 2)` / `(N, 3)` NumPy coordinate arrays to `pyproj.Transformer.transform` in a single vectorized call rather than vertex-by-vertex Python callbacks.

2. **`affine` `*` Multiplication (`#69`):**
   Composing or applying `affine.Affine` transforms with `*` in `RasterGrid` emits:
   ```text
   PendingDeprecationWarning: Use `@` matmul instead of `*` mul operator for matrix multiplication
   ```

3. **NumPy 2.0 `__array_wrap__` (`#47`):**
   NumPy 2.0 deprecated `__array_wrap__` implementations that do not accept `context` and `return_scalar` positionally. `Raster.__array_wrap__` in `rasters/raster.py` has already been updated to `def __array_wrap__(self, other, context=None, return_scalar=None, **kwargs)`, so regression tests are needed to verify zero warnings under NumPy 2.x and close `#47`.

---

## 2. Root-Cause Audit

1. **`rasters/vector_geometry.py` (`VectorGeometry.to_crs`):**
   Uses `shapely.ops.transform(Transformer.from_crs(...).transform, self.geometry)` and does not short-circuit when `self.crs == crs`.
2. **`rasters/raster_geometry.py` (`RasterGeometry.crosses_antimeridian`):**
   Imports `from shapely.ops import transform as shapely_transform` and calls `shapely_transform(to_polar, feature_shape)`.
3. **`rasters/raster_grid.py` (`Affine` matrix multiplication):**
   Uses `*` instead of `@` on `Affine` objects in seven locations:
   - `affine_center` (`self.affine * Affine.translation(0.5, 0.5)`)
   - `boundary` (`self.affine_center * (x_indices, y_indices)`)
   - `x_vector` (`self.affine_center * (...)`)
   - `y_vector` (`self.affine_center * (...)`)
   - `xy` (`self.affine_center * np.meshgrid(...)`)
   - `index_point` (`~self.affine_center * (x, y)`)
   - `shift_xy` (`self.affine * Affine.translation(...)`)
4. **`rasters/raster.py` (`Raster.__array_wrap__`):**
   Already updated with `(self, other, context=None, return_scalar=None, **kwargs)`; needs explicit ufunc regression coverage.

---

## 3. Development Plan

### Step 1: Vectorize `VectorGeometry.to_crs()` with `shapely.transform()` (`rasters/vector_geometry.py`)
Add `import numpy as np`, short-circuit identity CRS transformations (`self.crs == crs`), and replace `shapely.ops.transform()` with `shapely.transform(..., include_z=self.geometry.has_z)`:

```diff
--- a/rasters/vector_geometry.py
+++ b/rasters/vector_geometry.py
@@ -3,6 +3,7 @@ from __future__ import annotations
 from typing import Union
 
 import geopandas as gpd
+import numpy as np
 import shapely
 from pyproj import Transformer
 
@@ -66,9 +67,22 @@ class VectorGeometry(SpatialGeometry):
             VectorGeometry: The projected geometry.
         """
         crs = CRS(crs)
+
+        if self.crs == crs:
+            return self
+
+        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
+
+        def _transform_coords(coords: np.ndarray) -> np.ndarray:
+            if coords.shape[0] == 0:
+                return coords
+            x, y = transformer.transform(coords[:, 0], coords[:, 1])
+            if coords.shape[1] == 3:
+                return np.column_stack([x, y, coords[:, 2]])
+            return np.column_stack([x, y])
+
         result = self.contain(
-            shapely.ops.transform(Transformer.from_crs(self.crs, crs, always_xy=True).transform, self.geometry),
+            shapely.transform(self.geometry, _transform_coords, include_z=self.geometry.has_z),
             crs=crs)
         return result
```

### Step 2: Update `RasterGeometry.crosses_antimeridian` (`rasters/raster_geometry.py`)
Remove `from shapely.ops import transform as shapely_transform` and update `crosses_antimeridian` to use `shapely.transform()`:

```diff
--- a/rasters/raster_geometry.py
+++ b/rasters/raster_geometry.py
@@ -12,7 +12,6 @@ from pyproj import Transformer
 from rasterio.windows import Window
 from scipy.ndimage import shift
 from shapely.geometry.base import BaseGeometry
-from shapely.ops import transform as shapely_transform
 from .local_UTM_proj4 import local_UTM_proj4
 
 from .CRS import WGS84
@@ -529,7 +528,10 @@ class RasterGeometry(SpatialGeometry):
 
         feature_shape = self.boundary_latlon.geometry
         sign = 2. * (0.5 * (feature_shape.bounds[1] + feature_shape.bounds[3]) >= 0.) - 1.
-        polar_shape = shapely_transform(to_polar, feature_shape)
+        polar_shape = shapely.transform(
+            feature_shape,
+            lambda coords: np.column_stack(to_polar(coords[:, 0], coords[:, 1]))
+        )
 
         self._crosses_antimeridian = polar_shape.intersects(antimeridian_wedge)
```

### Step 3: Replace `Affine` `*` with `@` (`rasters/raster_grid.py`)
Update all `Affine` matrix multiplications and coordinate transformations in `RasterGrid` to use the `@` operator:

```diff
--- a/rasters/raster_grid.py
+++ b/rasters/raster_grid.py
@@ -303,7 +303,7 @@ class RasterGrid(RasterGeometry):
         Returns:
             Affine: Affine transform object.
         """
-        return self.affine * Affine.translation(0.5, 0.5)
+        return self.affine @ Affine.translation(0.5, 0.5)
 
     @property
     def cell_width(self) -> float:
@@ -435,7 +435,7 @@ class RasterGrid(RasterGeometry):
             ]))
 
         y_indices, x_indices = self.boundary_indices
-        x_boundary, y_boundary = self.affine_center * (x_indices, y_indices)
+        x_boundary, y_boundary = self.affine_center @ (x_indices, y_indices)
         points = np.c_[x_boundary, y_boundary]
         boundary = Polygon(points, crs=self.crs)
 
@@ -541,7 +541,7 @@ class RasterGrid(RasterGeometry):
         Returns:
             np.ndarray: Array of x-coordinates.
         """
-        return (self.affine_center * (np.arange(self.cols), np.full(self.cols, 0, dtype=np.float32)))[0]
+        return (self.affine_center @ (np.arange(self.cols), np.full(self.cols, 0, dtype=np.float32)))[0]
 
     @property
     def y_vector(self) -> np.ndarray:
@@ -551,7 +551,7 @@ class RasterGrid(RasterGeometry):
         Returns:
             np.ndarray: Array of y-coordinates.
         """
-        return (self.affine_center * (np.full(self.rows, 0, dtype=np.float32), np.arange(self.rows)))[1]
+        return (self.affine_center @ (np.full(self.rows, 0, dtype=np.float32), np.arange(self.rows)))[1]
 
     @property
     def xy(self) -> Tuple[np.ndarray, np.ndarray]:
@@ -561,7 +561,7 @@ class RasterGrid(RasterGeometry):
         Returns:
             Tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates.
         """
-        return self.affine_center * np.meshgrid(np.arange(self.cols), np.arange(self.rows))
+        return self.affine_center @ np.meshgrid(np.arange(self.cols), np.arange(self.rows))
 
     def index_point(self, point: Point) -> Tuple[int, int]:
         """
@@ -576,7 +576,7 @@ class RasterGrid(RasterGeometry):
         native_point = point.to_crs(self.crs)
         x = native_point.x
         y = native_point.y
-        col, row = ~self.affine_center * (x, y)
+        col, row = ~self.affine_center @ (x, y)
         col = int(round(col))
         row = int(round(row))
         index = (row, col)
@@ -668,7 +668,7 @@ class RasterGrid(RasterGeometry):
         Returns:
             RasterGrid: A new RasterGrid with the shift applied.
         """
-        new_affine = self.affine * Affine.translation(x_shift / self.cell_size, y_shift / self.cell_size)
+        new_affine = self.affine @ Affine.translation(x_shift / self.cell_size, y_shift / self.cell_size)
         grid = RasterGrid.from_affine(new_affine, self.rows, self.cols, self.crs)
 
         return grid
```

### Step 4: Add Regression Tests (`tests/test_vector_geometry.py`)
Create `tests/test_vector_geometry.py` to cover:
- `Point.to_crs()` (2D and 3D Z-preservation), `Polygon.to_crs()`, and empty `MultiPoint.to_crs()` (`#69` Shapely deprecation)
- `RasterGrid.crosses_antimeridian`, `RasterGrid.xy`, `RasterGrid.boundary`, `RasterGrid.index_point()`, and `RasterGrid.shift_xy()` (`#69` Shapely & `affine` `@` deprecations)
- NumPy ufuncs (`np.cos(raster)`, `np.sin(raster)`) on `Raster` (`#47` `__array_wrap__` deprecation)

```python
import warnings
import numpy as np
from rasters import Point, MultiPoint, Polygon, Raster, RasterGrid, WGS84


def _assert_no_deprecations(caught):
    deprecations = [
        w for w in caught
        if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
    ]
    assert len(deprecations) == 0, [str(w.message) for w in deprecations]


def test_point_to_crs_no_deprecation_warning():
    point = Point(-118.5, 34.5, crs=WGS84)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = point.to_crs("EPSG:32611")

    _assert_no_deprecations(caught)
    assert projected.crs.to_epsg() == 32611


def test_point_3d_to_crs_preserves_z():
    point = Point(-118.5, 34.5, 100.0, crs=WGS84)
    projected = point.to_crs("EPSG:32611")

    assert projected.geometry.has_z
    assert np.isclose(projected.geometry.z, 100.0)


def test_polygon_to_crs_no_deprecation_warning():
    polygon = Polygon(
        [(-119.0, 34.0), (-118.0, 34.0), (-118.0, 35.0), (-119.0, 35.0), (-119.0, 34.0)],
        crs=WGS84,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = polygon.to_crs("EPSG:32611")

    _assert_no_deprecations(caught)
    assert projected.crs.to_epsg() == 32611


def test_empty_multipoint_to_crs_no_deprecation_warning():
    mp = MultiPoint(crs=WGS84)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = mp.to_crs("EPSG:32611")

    _assert_no_deprecations(caught)
    assert projected.geometry.is_empty
    assert projected.crs.to_epsg() == 32611


def test_raster_grid_operations_no_deprecation_warning():
    grid = RasterGrid.from_bbox(
        xmin=-119.0, ymin=34.0, xmax=-118.0, ymax=35.0, cell_size=0.1, crs=WGS84
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crosses = grid.crosses_antimeridian
        _ = grid.xy
        _ = grid.x_vector
        _ = grid.y_vector
        _ = grid.boundary
        _ = grid.index_point(Point(-118.5, 34.5, crs=WGS84))
        _ = grid.shift_xy(0.1, -0.1)

    _assert_no_deprecations(caught)
    assert crosses is False


def test_raster_numpy_ufunc_no_deprecation_warning():
    grid = RasterGrid.from_bbox(
        xmin=-119.0, ymin=34.0, xmax=-118.0, ymax=35.0, cell_size=0.5, crs=WGS84
    )
    raster = Raster(np.linspace(0, np.pi, 4).reshape(2, 2), geometry=grid)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = np.cos(raster) + np.sin(raster)

    _assert_no_deprecations(caught)
    assert isinstance(result, Raster)
    assert result.shape == (2, 2)
```

---

## 4. Verification & Release Plan

1. **Run test suite with deprecation warnings escalated to errors:**
   ```bash
   pytest -W error::DeprecationWarning -W error::PendingDeprecationWarning
   ```
2. **Confirm `pyproject.toml` version and Python floor:**
   - `version = "1.21.0"`
   - `requires-python = ">=3.11"`
3. **Commit, tag, and publish release `v1.21.0`** to PyPI via the GitHub Release workflow, then close issues `#69` and `#47`:
   ```bash
   gh issue close 69 47 --comment "Resolved in v1.21.0."
   ```
````[cite: 2, 3]