import sys
import os
import json
from common import utils
from alignment.optimize_mesh import optimize_meshes
import math
import numpy as np
from scipy import spatial
import common.trans_models as models

overall_args = utils.load_json_file('arguments/overall_args.json')
log_controller = utils.LogController('alignment', 'optimize_3d', os.path.join(overall_args["base"]["workspace"], 'log'))

SAMPLED_POINTS_NUM = 50


def read_layer_from_file(tilespecs_file_path):
    layer = None
    with open(tilespecs_file_path, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        if tile['layer'] is None:
            log_controller.debug("Error reading layer in one of the tiles in: {0}".format(tilespecs_file_path))
            sys.exit(1)
        if layer is None:
            layer = tile['layer']
        if layer != tile['layer']:
            log_controller.debug("Error when reading tiles from {0} found inconsistent "
                                 "layers numbers: {1} and {2}".format(tilespecs_file_path, layer, tile['layer']))
            sys.exit(1)
    if layer is None:
        log_controller.error("Error reading layers file: {0}. "
                                          "No layers found.".format(tilespecs_file_path))
        raise ValueError
    return int(layer)


def compute_points_model_halo(url_optimized_mesh0, points_tree):
    log_controller.debug("Computing Points Transform Model Halo")

    # Sample SAMPLED_POINTS_NUM points to find the closest neighbors to these points
    sampled_points_indices = np.random.choice(url_optimized_mesh0.shape[0], SAMPLED_POINTS_NUM, replace=False)
    sampled_points = url_optimized_mesh0[np.array(sampled_points_indices)]

    # Find the minimal distance between the sampled points to any other point, by finding the closest point to each
    # point and take the minimum among the distances
    distances, _ = points_tree.query(sampled_points, 2)
    min_point_dist = np.min(distances[:, 1])
    halo = 2 * min_point_dist
    log_controller.debug("Points model halo: {}".format(halo))

    return halo


def get_points_transform_model(url_optimized_mesh, bbox, pts_tree, halo):
    # Find the tile bbox with a halo around it
    bbox_with_halo = list(bbox)
    bbox_with_halo[0] -= halo
    bbox_with_halo[2] -= halo
    bbox_with_halo[1] += halo
    bbox_with_halo[3] += halo

    # filter the matches according to the new bounding box
    # (first pre-filter entire mesh using a halo of diagonal + 2*halo + 1) around the top-left point
    top_left = np.array([bbox[0], bbox[2]])
    bottom_right = np.array([bbox[1], bbox[3]])
    pre_filtered_indices = pts_tree.query_ball_point(top_left, np.linalg.norm(bottom_right - top_left) + 2 * halo + 1)
    if len(pre_filtered_indices) == 0:
        log_controller.debug("Could not find any mesh points in bbox {}, skipping the tile")
        return None

    filtered_src_points = []
    filtered_dest_points = []
    for p_src, p_dest in zip(url_optimized_mesh[0][np.array(pre_filtered_indices)],
                             url_optimized_mesh[1][np.array(pre_filtered_indices)]):
        if (bbox_with_halo[0] <= p_src[0] <= bbox_with_halo[1]) and (
                bbox_with_halo[2] <= p_src[1] <= bbox_with_halo[3]):
            filtered_src_points.append(p_src)
            filtered_dest_points.append(p_dest)

    if len(filtered_src_points) == 0:
        log_controller.debug("Could not find any mesh points in bbox {}, skipping the tile")
        return None

    # create the tile transformation
    model = models.PointsTransformModel((filtered_src_points, filtered_dest_points))
    return model


def compute_new_bounding_box(tile_ts):
    """Computes a bounding box given the tile's transformations (if any),
       and the new model to be applied last"""
    # We must have a non-affine transformation, so compute the transformation of all the boundary pixels
    # using a forward transformation from the boundaries of the source image to the destination
    # Assumption: There won't be a pixel inside an image that goes out of the boundary
    boundary1 = np.array([[float(p), 0.] for p in np.arange(tile_ts["width"])])
    boundary2 = np.array([[float(p), float(tile_ts["height"] - 1)] for p in np.arange(tile_ts["width"])])
    boundary3 = np.array([[0., float(p)] for p in np.arange(tile_ts["height"])])
    boundary4 = np.array([[float(tile_ts["width"] - 1), float(p)] for p in np.arange(tile_ts["height"])])
    boundaries = np.concatenate((boundary1, boundary2, boundary3, boundary4))

    for modelspec in tile_ts.get("transforms", []):
        model = models.Transforms.from_tilespec(modelspec)
        boundaries = model.apply(boundaries)

    # Find the bounding box of the boundaries
    min_xy = np.min(boundaries, axis=0)
    max_xy = np.max(boundaries, axis=0)
    # Rounding to avoid float precision errors due to representation
    new_bbox = [int(math.floor(round(min_xy[0], 5))), int(math.ceil(round(max_xy[0], 5))),
                int(math.floor(round(min_xy[1], 5))), int(math.ceil(round(max_xy[1], 5)))]
    return new_bbox


def save_optimized_mesh(tilespecs_file_path, url_optimized_mesh, output_folder_path):
    log_controller.debug("Working on:{}".format(tilespecs_file_path))

    # Use the first tile to find the halo for the entire section
    points_tree = spatial.KDTree(url_optimized_mesh[0])
    halo = compute_points_model_halo(url_optimized_mesh[0], points_tree)

    tilespecs_basename = os.path.basename(tilespecs_file_path)
    output_file_path = os.path.join(output_folder_path, tilespecs_basename)
    # read tilespec
    with open(tilespecs_file_path, 'r') as data_file:
        data = json.load(data_file)

    if len(data) > 0:
        tiles_to_remove = []
        # change the transformation
        for tile_index, tile in enumerate(data):
            # Create the PointsTransformModel for the current tile
            tile_model = get_points_transform_model(url_optimized_mesh, tile["bbox"], points_tree, halo)
            if tile_model is None:
                tiles_to_remove.append(tile_index)
            else:
                # Add the model to the tile
                tile_transform = tile_model.to_modelspec()
                tile.get("transforms", []).append(tile_transform)
                # Compute new bounding box
                tile["bbox"] = compute_new_bounding_box(tile)

        for tile_index in sorted(tiles_to_remove, reverse=True):
            log_controller.debug("Removing tile {} "
                                 "from {}".format(data[tile_index]["mipmapLevels"]["0"]["imageUrl"], output_file_path))
            del data[tile_index]

        # save the output tile spec
        utils.save_json_file(output_file_path, data)
    else:
        log_controller.debug('Nothing to write for tilespec {}'.format(tilespecs_file_path))


def read_ts_layers(tile_files):
    tilespecs_file_to_layer_id = {}
    log_controller.debug("Reading tilespec files...")
    # TODO - make sure its not a json files list
    with open(tile_files[0], 'r') as f:
        actual_tile_urls = [line.strip('\n') for line in f.readlines()]

    for url in actual_tile_urls:
        file_name = url.replace('file://', '')
        layer_id = read_layer_from_file(file_name)
        tilespecs_file_basename = os.path.basename(url)
        tilespecs_file_to_layer_id[tilespecs_file_basename] = layer_id

    return tilespecs_file_to_layer_id, actual_tile_urls


def optimize_layers_elastic(tile_files, corr_files, out_dir, conf=None):
    tilespecs_file_to_layer_id, all_tile_urls = read_ts_layers(tile_files)

    # TODO: the tile_files order should imply the order of sections

    # TODO - make sure its not a json files list
    with open(corr_files[0], 'r') as f:
        actual_corr_files = [line.replace('file://', '').strip('\n') for line in f.readlines()]

    conf_dict = {}
    hex_spacing = 1500  # default value (from block matching)
    if conf is not None:
        with open(conf, 'r') as f:
            params = json.load(f)
            conf_dict = params["OptimizeLayersElastic"]
            hex_spacing = params["MatchLayersBlockMatching"]["hex_spacing"]

    # Create a per-layer optimized mesh
    optimized_meshes = optimize_meshes(actual_corr_files, hex_spacing, conf_dict)

    # Save the output
    utils.create_dir(out_dir)
    for ts_url in all_tile_urls:
        tilespecs_file_path = ts_url.replace('file://', '')
        save_optimized_mesh(tilespecs_file_path, optimized_meshes[tilespecs_file_path], out_dir)
    log_controller.debug("Optimization done.")
