# `rasters` python package

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

This Python software package provides a comprehensive solution for handling gridded and swath raster data. It offers an object-oriented interface for ease of use, treating rasters, raster geometries, and vector geometries as objects. This allows for seamless map visualization in Jupyter notebooks and efficient resampling between swath and grid geometries.

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

### Does your work relate to current or future NASA (include reimbursable) work that has value to the conduct of aeronautical and space activities?  If so, please explain:

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
    
## Changelog

### 1.1.0

The `KDTree` class can now save to file with the `.save` method and load from file with the `KDTree.load` class method.