"""
This package manages the geographic information associated with data points in both swath and grid rasters.
"""
from __future__ import annotations

import io
import json
import locale
import logging
import math
import os

import warnings
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from os import makedirs
from os.path import dirname, exists, abspath, expanduser, splitext
from typing import List, Tuple, Iterator, Union, Any, Optional, Dict
import msgpack
import msgpack_numpy
import PIL
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import shapely
import skimage
import skimage.transform
from PIL import Image
from affine import Affine
from astropy.visualization import MinMaxInterval, AsymmetricPercentileInterval
from matplotlib import colors
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.core.umath import logical_or, isfinite
from pyproj import CRS, Transformer
from pyproj import transform
from pyproj._crs import _CRS

import pykdtree
from scipy.spatial import cKDTree   
from pyresample import SwathDefinition, AreaDefinition
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info
from rasterio import DatasetReader

from rasterio.enums import MergeAlg
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import shift
from scipy.spatial import cKDTree as KDTree
from shapely.geometry import Point, LinearRing, MultiPoint
from shapely.geometry.base import BaseGeometry, CAP_STYLE, JOIN_STYLE, geom_factory
from shapely.ops import transform as shapely_transform
from six import string_types

from .constants import *
from .CRS import CRS, WGS84
from .bbox import BBox
from .spatial_geometry import SpatialGeometry
from .coordinate_array import CoordinateArray
from .vector_geometry import VectorGeometry, SingleVectorGeometry, MultiVectorGeometry
from .point import Point
from .multi_point import MultiPoint
from .polygon import Polygon
from .multi_polygon import MultiPolygon
from .wrap_geometry import wrap_geometry
from .kdtree import KDTree
from .raster_geometry import RasterGeometry
from .raster_geolocation import RasterGeolocation
from .out_of_bounds_error import OutOfBoundsError
from .raster_grid import RasterGrid
from .raster import Raster
from .where import where
from .clip import clip

__author__ = "Gregory Halverson"

try:
    locale.setlocale(locale.LC_ALL, "")
except Exception as e:
    pass





