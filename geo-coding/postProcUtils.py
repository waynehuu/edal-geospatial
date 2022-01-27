import os
from glob import glob
import argparse
import multiprocessing

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from skimage import draw, measure, io
import cv2
from scipy.spatial import KDTree
from skimage.morphology import disk, dilation


def make_save_dir_name(args: argparse.Namespace):
    save_dir_name = ''
    args_dict = vars(args)
    for key, value in args_dict.items():
        if not key in ['results_dir', 'save_root', 'task_name', 'matching_file', 'n_cores']:
            save_dir_name = '_'.join(
                [save_dir_name, key, str(value).replace('.', 'p')])
    return save_dir_name[1:]


def rasterize_polygons(map_size, coords: list):
    dummy = np.zeros(map_size)
    if coords:
        for c in coords:
            c = np.array(c)
            xx, yy = draw.polygon(c[:, 0], c[:, 1])
            dummy[xx, yy] = 1
    return dummy


def dilated_map_overlay(in_map, dilated_category_polygons):
    out_map = np.multiply(
        rasterize_polygons(in_map.shape, dilated_category_polygons),
        in_map
    )
    return out_map


def conf_to_pred(in_map: np.ndarray, min_conf, min_area):
    ''' Make predictions from confidence maps
    conf: confidence map, numpy array
    '''
    props = []
    if min_conf <= in_map.min():
        out_map = np.ones(in_map.shape)
    elif min_conf > in_map.max():
        out_map = np.zeros(in_map.shape)
    else:
        candiate_mask = in_map >= min_conf
        label = measure.label(candiate_mask)
        props = measure.regionprops(label, in_map)

        dummy = np.zeros(in_map.shape)
        for p in props:
            if p.area >= min_area:
                for x, y in p.coords:
                    dummy[x, y] = 1
        out_map = dummy

    return props, out_map


def dilate(in_map, kernel_size=5):
    if kernel_size > 0:
        selem = disk(kernel_size)
        return dilation(in_map, selem)
    else:
        return in_map


def make_polygons(in_map, epsilon=2):
    polygons, _ = cv2.findContours(in_map.astype(
        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if polygons:
        approx_polygons = [measure.approximate_polygon(
            np.squeeze(p), tolerance=epsilon) for p in polygons if len(p) > 1]
        return approx_polygons
    else:
        return polygons


def res_com_classify(in_map, commercial_area_threshold, dilated_in_map=None, kernel_size=5):
    ''' Classify pixel groups into residential/commercial based on dilated area
    in_map: binary prediction map
    '''
    if dilated_in_map is None:
        dilated_in_map = dilate(in_map, kernel_size)

    # dilated_polygons = make_polygons(dilated_in_map) # get list of polygon vertices (shape=(n, 2))
    # dilated_commercial_polygons, dilated_residential_polygons = \
    #     [], []
    
    # if not dilated_polygons: # when the vertices list is empty
    #     pass
    # else:
    #     for p in dilated_polygons:
    #         ps = Polygon(p)
    #         # append polygons vertices to corresponding list based on area thresholding
    #         if ps.area >= commercial_area_threshold:
    #             dilated_commercial_polygons.append(p)
    #         else:
    #             dilated_residential_polygons.append(p)
                
    # # commercial_map = dilated_map_overlay(in_map, dilated_commercial_polygons)
    # # residential_map = dilated_map_overlay(in_map, dilated_residential_polygons)

    # commercial_map = rasterize_polygons(in_map.shape, dilated_commercial_polygons)
    # residential_map = rasterize_polygons(in_map.shape, dilated_residential_polygons)
    # out_map = residential_map + 2 * commercial_map # 1 for residential, 2 for commercial

    label = measure.label(dilated_in_map)
    props = measure.regionprops(label, dilated_in_map)

    dummy = np.zeros(in_map.shape)
    if props:
        for p in props:
            rescom = 2 \
                if p.area >= commercial_area_threshold else 1 # 1 for residential, 2 for commercial
            for x, y in p.coords:
                    dummy[x, y] = rescom
    
    return dummy


def object_wise_group(in_map, dilated_in_map=None, kernel_size=5):
    ''' Group nearby polygons into object groups based on dilated area
    in_map: binary prediction map, not dilated
    dilated_in_map (optional): if provided, should be the dilated in_map
    '''
    if dilated_in_map is None:
        dilated_in_map = dilate(in_map, kernel_size)

    label = measure.label(dilated_in_map)
    props = measure.regionprops(label, in_map)

    dummy = np.zeros(in_map.shape)
    if props:
        for i, p in enumerate(props):
            for x, y in p.coords:
                    dummy[x, y] = i + 1 # each group will be given an interger as an indicator
    # out_map = np.multiply(in_map, dummy) # remove extra dilated pixels

    return dummy

