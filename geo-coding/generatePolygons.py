'''
Steps in post-processing:
1. Create binary masks from confidence maps
2. Use dilation-based method to classify pixel groups into residential/commercial
3. Use dilation-based method to group pixels into objects
4. Vectorize pixel group to vector polygons
'''

import os
from glob import glob
import multiprocessing
import argparse
from datetime import datetime

from tqdm import tqdm

from skimage import measure, io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio import features
from shapely.geometry import Polygon

import tasks.polygonGeneration.postProcUtils as pp

################## Parameters

conf_dir = '/home/wh145/results/solarmapper/ca_only/ec_1e-3_dc_1e-2/cls_wt_1_1/all_ct/ecresnet50_dcunet_dsct_new_non_random_3_splits_lre1e-03_lrd1e-02_ep180_bs7_ds30_dr0p1_crxent7p0_softiou3p0'
output_dir = '/home/wh145/mrs/tasks/polygonGeneration/results/final/'
shp = gpd.read_file('/home/wh145/mrs/tasks/DeliveryGrid2016/DeliveryGrid.shp')
min_area = 10
commercial_area_threshold = 6000
rescom_kernel_size = 10
object_kernel_size = 10

in_dim = 2541
out_dim = 2500

##################

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--min_conf', type=float, default=0.5)
parser.add_argument('--min_conf_path', type=str, default=None)
parser.add_argument('--tolerance', type = float, default=0.5)
parser.add_argument('--session_name', type = str, default=None)
parser.add_argument('--tile_list_path', type=str, default=None)
super_args = parser.parse_args()

##################

def register_to_tile(row, shapefile=shp, in_dim=in_dim, out_dim=out_dim):

    tile_name = row['tile_name']
    polygon_xys = row['coords']
    origin_x = float(
        shapefile[shapefile['NAME'].str.contains(tile_name)]['UPPER_LE_X'])
    origin_y = float(
        shapefile[shapefile['NAME'].str.contains(tile_name)]['LOWER_RI_Y'])
    scale_fact = out_dim / in_dim

    output_coords = [(x * scale_fact + origin_x, (in_dim - y) * scale_fact + origin_y)
                     for x, y in polygon_xys]

    return Polygon(output_coords)

##################

if super_args.min_conf_path is not None:
    min_conf = np.load(super_args.min_conf_path) * 255
else:
    min_conf = super_args.min_conf * 255

all_tiles = glob(os.path.join(conf_dir, '*.png'))

if super_args.tile_list_path is not None:
    tile_list = pd.read_csv(super_args.tile_list_path)
    tile_list = tile_list['tile_name'].to_list()
    tile_list = [t for t in all_tiles if os.path.basename(t).replace('_conf.png', '') in tile_list]
else:
    tile_list = all_tiles


tile_name_list = [os.path.basename(s).replace('_conf.png', '') for s in tile_list]
tile_conf_list = [io.imread(s) for s in tqdm(tile_list, desc='Loading conf maps...')]

polygons_info = [] # Column names: coords, area_px, rescom, object_id, tile_name

for tile_name, conf_map in tqdm(zip(tile_name_list, tile_conf_list), desc='Processing each tile...'):

    props, pred_map = pp.conf_to_pred(conf_map, min_conf, min_area)
    rescom_map = pp.res_com_classify(pred_map, commercial_area_threshold, kernel_size=rescom_kernel_size)
    object_map = pp.object_wise_group(pred_map, kernel_size=object_kernel_size)

    label = measure.label(pred_map)
    pred_props = measure.regionprops(label, pred_map)
    rescom_props = measure.regionprops(label, rescom_map)
    object_props = measure.regionprops(label, object_map)

    for p, r, o in zip(pred_props, rescom_props, object_props):
        polygons_info.append(
            (
                [pair + p.bbox[:2][::-1] for pair in list(
                    pp.make_polygons(p.image, epsilon=super_args.tolerance)[0])],
                p.area,
                'residential' if r.max_intensity == 1 else 'commercial',
                '{}_{}'.format(tile_name, str(int(o.max_intensity))),
                tile_name
            )
        )

polygons_pd = pd.DataFrame(
    polygons_info, 
    columns=['coords', 'area_px', 'use_type', 'object_id', 'tile_name']
)

config_name = 'min_conf_{}_tolerance_{}'.format(
    str(min_conf).replace('.', 'p')[:4],
    str(super_args.tolerance).replace('.', 'p')[:4]
    )

if super_args.session_name is not None:
    config_name = '_'.join((super_args.session_name, config_name))

output_path = os.path.join(
    output_dir, 
    '_'.join(
        (datetime.now().strftime(r'%m%d%y_%H%M'), config_name)
        )
    )

print('Converting relative coordinate to latlon coordinates...')
polygons_pd['geometry'] = polygons_pd.apply(register_to_tile, axis=1)

output_polygons = gpd.GeoDataFrame(polygons_pd.drop('coords', axis=1), crs='EPSG:2234', geometry=polygons_pd['geometry'])
output_polygons['area_sqm'] = output_polygons.area * 0.092903

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
polygons_save_path = os.path.join(
    output_path,
    '{}_polygons.geojson'.format(config_name)
)

centroids_save_path = os.path.join(
    output_path,
    '{}_centroids.geojson'.format(config_name)
)

output_polygons.to_file(polygons_save_path.replace('.geojson', '_xy.geojson'), driver='GeoJSON')
output_polygons.to_file(
    polygons_save_path.replace('.geojson', '_xy.shp'))

output_polygons_latlon = output_polygons.to_crs(epsg=4326)
output_polygons_latlon.to_file(polygons_save_path.replace('.geojson', '_latlon.geojson'), driver='GeoJSON')
output_polygons_latlon.to_file(
    polygons_save_path.replace('.geojson', '_latlon.shp'))

# output_centroids = output_polygons.copy()
# output_centroids['geometry'] = output_polygons['geometry'].centroid
# output_centroids.to_file(centroids_save_path, driver='GeoJSON')
