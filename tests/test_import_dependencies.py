import pytest

# List of dependencies
dependencies = [
    "affine",
    "astropy",
    "geopandas",
    "h5py",
    "matplotlib",
    "msgpack",
    "msgpack_numpy",
    "numpy",
    "pandas",
    "PIL",
    "pykdtree",
    "pyproj",
    "pyresample",
    "pytest",
    "rasterio",
    "scipy",
    "shapely",
    "six",
    "skimage",
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
