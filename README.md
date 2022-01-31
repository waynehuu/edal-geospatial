# Energy Data Analytics Lab's tutorials on geo-spatial operations

## Table of contents
- [Geospatial set operations](#geospatial-set-operations)
- [Geospatial plotting](#geospatial-plotting)
- [Geo-coding](#geo-coding)
- [Rasterization and vectorization](#rasterization-and-vectorization)

## Geospatial set operations
### Find the intersection/union/difference between two geospatial dataset and output a geospatial file
### Useful package: 
- `geopandas`: [Set-Operations with Overlay](https://geopandas.org/en/stable/docs/user_guide/set_operations.html)
### Example: 
- [Find out how many building there are in each image tile](https://github.com/waynehuu/mrs/blob/publish-solarmapper/solarmapper_demo/building_density_tile_stratified_sampling/building_density_stratified_sampling.ipynb)

## Geospatial plotting
### Plot points/polygons/any other shapes on a map
### Useful package: 
- `geopandas` 
- `cartopy` 
- `matplotlib's Basemap`: not recommended, has been depreciated and replaced by `cartopy`
### Example:
- [Plot sampling locations of Connecticut annotated tiles on Connecticut town-level map](geospatial-plotting/geospatial-plotting.ipynb)
- [Plot density map of solar PV panel area at any resolution](https://github.com/energydatalab/mrs/blob/main/solarmapper_demo/results_eval_and_viz/ct_municipality_level_analysis.ipynb)

## Geo-coding
### Use case: Convert pixel coordinates of polygons/points/lines to geospatial coordinates
### Useful package: 
- `geopandas`

### Example:
- [Convert polygon's pixel coordinates to geospatial coordinates](geo-coding/geo_coding.ipynb)
- `register_to_tile()` in [`generatePolygons.py`](geo-coding/generatePolygons.py)

## Rasterization and vectorization
### Convert between vector shapes and raster images
### Useful package: 
- `geopandas`
- `rasterio`
- `opencv`
- `skimage`
### Example:
#### Rasterization:
- `pyimannotate`'s [binarymask.py](https://github.com/energydatalab/pyimannotate/blob/master/binarymask.py): converts manual polygon annotations to binary raster masks
- [Script for rasterizing GeoTIFF images](vectorization-rasterization/preprocess.py)

#### vectorization:
- `make_polygons()` in [`posrProcUtils.py`](geo-coding/generatePolygons.py)
- `opencv`'s [`cv2.findContours`](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0)
- Usually coupled with a polygon approximation process based on the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) to reduce the number of vertices. E.g. [`skimage.measure.approximate_polygon()`](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.approximate_polygon).

## Managing projections
Geospatial datasets sometimes come with different coordinates reference systems (CRS). Convert them to the same CRS before making any overlay analysis or plotting.

### Useful package
- `geopandas`

### Example:
-  [Managing Projections](https://geopandas.org/en/stable/docs/user_guide/projections.html) with `geopandas`
