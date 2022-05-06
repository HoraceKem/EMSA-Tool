import sys
import os
import argparse
import json
from common import utils
from optimize_mesh import optimize_meshes
import math
import numpy as np
from scipy import spatial
import multiprocessing as mp
import common.trans_models as models

overall_args = utils.load_json_file('../arguments/overall_args.json')
utils.create_dir(overall_args["base"]["workspace"])
log_controller = utils.LogController('alignment', os.path.join(overall_args["base"]["workspace"], 'log'),
                                     overall_args["base"]["running_mode"])

SAMPLED_POINTS_NUM = 50


def read_layer_from_file(tiles_spec_fname):
    layer = None
    with open(tiles_spec_fname, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        if tile['layer'] is None:
            print("Error reading layer in one of the tiles in: {0}".format(tiles_spec_fname))
            sys.exit(1)
        if layer is None:
            layer = tile['layer']
        if layer != tile['layer']:
            print("Error when reading tiles from {0} found inconsistent layers numbers: {1} and {2}".format(tiles_spec_fname, layer, tile['layer']))
            sys.exit(1)
    if layer is None:
        print("Error reading layers file: {0}. No layers found.".format(tiles_spec_fname))
        sys.exit(1)
    return int(layer)


def compute_points_model_halo(url_optimized_mesh0, points_tree):
    log_controller.debug("Computing Points Transform Model Halo")

    # Sample SAMPLED_POINTS_NUM points to find the closest neighbors to these points
    sampled_points_indices = np.random.choice(url_optimized_mesh0.shape[0], SAMPLED_POINTS_NUM, replace=False)
    sampled_points = url_optimized_mesh0[np.array(sampled_points_indices)]

    # Find the minimal distance between the sampled points to any other point, by finding the closest point to each point
    # and take the minimum among the distances
    distances, _ = points_tree.query(sampled_points, 2)
    min_point_dist = np.min(distances[:,1])
    halo = 2 * min_point_dist
    log_controller.debug("Points model halo: {}".format(halo))

    return halo


def get_points_transform_model(url_optimized_mesh, bbox, points_tree, halo):
    # Find the tile bbox with a halo around it
    bbox_with_halo = list(bbox)
    bbox_with_halo[0] -= halo
    bbox_with_halo[2] -= halo
    bbox_with_halo[1] += halo
    bbox_with_halo[3] += halo

    # filter the matches according to the new bounding box
    # (first pre-filter entire mesh using a halo of "diagonal + 2*halo + 1) around the top-left point)
    top_left = np.array([bbox[0], bbox[2]])
    bottom_right = np.array([bbox[1], bbox[3]])
    pre_filtered_indices = points_tree.query_ball_point(top_left, np.linalg.norm(bottom_right - top_left) + 2 * halo + 1)
    #print bbox_with_halo, "with filtered_indices:", pre_filtered_indices

    if len(pre_filtered_indices) == 0:
        log_controller.debug("Could not find any mesh points in bbox {}, skipping the tile")
        return None

    filtered_src_points = []
    filtered_dest_points = []
    for p_src, p_dest in zip(url_optimized_mesh[0][np.array(pre_filtered_indices)], url_optimized_mesh[1][np.array(pre_filtered_indices)]):
        if (bbox_with_halo[0] <= p_src[0] <= bbox_with_halo[1]) and (bbox_with_halo[2] <= p_src[1] <= bbox_with_halo[3]):
            filtered_src_points.append(p_src)
            filtered_dest_points.append(p_dest)

    if len(filtered_src_points) == 0:
        log_controller.debug("Could not find any mesh points in bbox {}, skipping the tile")
        return None

    # print bbox_with_halo, "with pre_filtered_indices len:", len(pre_filtered_indices), "with matches_str:", matches_str
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
    min_XY = np.min(boundaries, axis=0)
    max_XY = np.max(boundaries, axis=0)
    # Rounding to avoid float precision errors due to representation
    new_bbox = [int(math.floor(round(min_XY[0], 5))), int(math.ceil(round(max_XY[0], 5))), int(math.floor(round(min_XY[1], 5))), int(math.ceil(round(max_XY[1], 5)))]
    return new_bbox


def save_json_file(out_fname, data):
    with open(out_fname, 'w') as outjson:
        json.dump(data, outjson, sort_keys=True, indent=4)
        print('Wrote tilespec to {0}'.format(out_fname))
        sys.stdout.flush()


def save_optimized_mesh(ts_fname, url_optimized_mesh, out_dir):
    log_controller.debug("Working on:{}".format(ts_fname))

    # Use the first tile to find the halo for the entire section
    points_tree = spatial.KDTree(url_optimized_mesh[0])
    halo = compute_points_model_halo(url_optimized_mesh[0], points_tree)

    ts_base = os.path.basename(ts_fname)
    out_fname = os.path.join(out_dir, ts_base)
    # read tilespec
    data = None
    with open(ts_fname, 'r') as data_file:
        data = json.load(data_file)

    if len(data) > 0:
        tiles_to_remove = []
        # change the transfromation
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
                                 "from {}".format(data[tile_index]["mipmapLevels"]["0"]["imageUrl"], out_fname))
            del data[tile_index]

        # save the output tile spec
        save_json_file(out_fname, data)
    else:
        print('Nothing to write for tilespec {}'.format(ts_fname))
        sys.stdout.flush()


def save_optimized_meshes(all_tile_urls, optimized_meshes, out_dir, processes_num=1):
    # Do the actual multiprocessed save
    pool = mp.Pool(processes=processes_num)
    print("Using {} processes to save the output jsons".format(processes_num))

    all_results = []
    for ts_url in all_tile_urls:
        ts_fname = ts_url.replace('file://', '')
        res = pool.apply_async(save_optimized_mesh, (ts_fname, optimized_meshes[ts_fname], out_dir))
        all_results.append(res)

    for res in all_results:
        res.get()

    pool.close()
    pool.join()
        
def read_ts_layers(tile_files):
    tsfile_to_layerid = {}

    log_controller.debug("Reading tilespec files...")

    # TODO - make sure its not a json files list
    actual_tile_urls = []
    with open(tile_files[0], 'r') as f:
        actual_tile_urls = [line.strip('\n') for line in f.readlines()]
    
    for url in actual_tile_urls:
        file_name = url.replace('file://', '')
        layerid = read_layer_from_file(file_name)
        tsfile = os.path.basename(url)
        tsfile_to_layerid[tsfile] = layerid

    return tsfile_to_layerid, actual_tile_urls
    


def optimize_layers_elastic(tile_files, corr_files, out_dir, max_layer_distance, conf=None, skip_layers=None, threads_num=4):

    tsfile_to_layerid, all_tile_urls = read_ts_layers(tile_files)

    # TODO: the tile_files order should imply the order of sections

    # TODO - make sure its not a json files list
    actual_corr_files = []
    with open(corr_files[0], 'r') as f:
        actual_corr_files = [line.replace('file://', '').strip('\n') for line in f.readlines()]

    conf_dict = {}
    hex_spacing = 1500 # default value (from block matching)
    if conf is not None:
        with open(conf, 'r') as f:
            params = json.load(f)
            conf_dict = params["OptimizeLayersElastic"]
            hex_spacing = params["MatchLayersBlockMatching"]["hex_spacing"]

    print(actual_corr_files)
    # Create a per-layer optimized mesh
    optimized_meshes = optimize_meshes(actual_corr_files, hex_spacing, conf_dict)
    
    # Save the output
    utils.create_dir(out_dir)
    save_optimized_meshes(all_tile_urls, optimized_meshes, out_dir, threads_num)

    print("Done.")
    


def main():
    print(sys.argv)

    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('--tile_files', metavar='tile_files', type=str, nargs='+', required=True,
                        help='the list of tile spec files to align')
    parser.add_argument('--corr_files', metavar='corr_files', type=str, nargs='+', required=True,
                        help='the list of corr spec files that contain the matched layers')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory that will include the aligned sections tiles (default: .)',
                        default='./')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)
    parser.add_argument('-s', '--skip_layers', type=str, 
                        help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
                        default=None)
    parser.add_argument('-d', '--max_layer_distance', type=int, 
                        help='the largest distance between two layers to be matched (default: 1)',
                        default=1)


    args = parser.parse_args()

    print("tile_files: {0}".format(args.tile_files))
    print("corr_files: {0}".format(args.corr_files))

    optimize_layers_elastic(args.tile_files, args.corr_files,
                            args.output_dir, args.max_layer_distance,
                            skip_layers=args.skip_layers, threads_num=args.threads_num)


if __name__ == '__main__':
    main()

