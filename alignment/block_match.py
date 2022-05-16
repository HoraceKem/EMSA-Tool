import os
import numpy as np
import json
from scipy.spatial import distance
from scipy import spatial
from common import utils
from common.bounding_box import BoundingBox
from alignment import PMCC_filter
import multiprocessing as mp
from renderer.tilespec_affine_renderer import TilespecAffineRenderer

overall_args = utils.load_json_file('arguments/overall_args.json')
log_controller = utils.LogController('alignment', 'block', os.path.join(overall_args["base"]["workspace"], 'log'))


def get_mfov_centers_from_json(indexed_tilespecs: dict) -> dict:
    """
    Get mfov centers
    :param indexed_tilespecs:
    :return:
    """
    mfov_centers = {}
    for mfov in indexed_tilespecs.keys():
        mfov_tiles = indexed_tilespecs[mfov].values()
        tile_bboxes = list(zip(*[tile["bbox"] for tile in mfov_tiles]))
        min_x = min(tile_bboxes[0])
        max_x = max(tile_bboxes[1])
        min_y = min(tile_bboxes[2])
        max_y = max(tile_bboxes[3])
        mfov_centers[mfov] = np.array([(min_x / 2.0 + max_x / 2.0, min_y / 2.0 + max_y / 2.0)])
    return mfov_centers


def get_best_transformations(pre_mfov_matches: dict, tilespecs_file_path1: str, tilespecs_file_path2: str,
                             mfov_centers1: dict, mfov_centers2: dict, sorted_mfovs1):
    """
    Returns a dictionary that maps a mfov number to a matrix that best describes the transformation to
    the other section. As not all mfov's may be matched, some mfovs will be missing from the dictionary.
    If the given tiles file names are reversed_flag, an inverted matrix is returned
    :param pre_mfov_matches:
    :param tilespecs_file_path1:
    :param tilespecs_file_path2:
    :param mfov_centers1:
    :param mfov_centers2:
    :param sorted_mfovs1:
    :return:
    """
    transforms = {}
    if tilespecs_file_path1 == pre_mfov_matches["tilespec1"] and tilespecs_file_path2 == pre_mfov_matches["tilespec2"]:
        reversed_flag = False
    elif tilespecs_file_path1 == pre_mfov_matches["tilespec2"] and \
            tilespecs_file_path2 == pre_mfov_matches["tilespec1"]:
        reversed_flag = True
    else:
        log_controller.error("Error: could not find pre_matches between tilespecs"
                             " {} and {} (found tilespecs {} and "
                             "{} instead!).".format(tilespecs_file_path1, tilespecs_file_path2,
                                                    pre_mfov_matches["tilespec1"], pre_mfov_matches["tilespec2"]))
        return {}

    if reversed_flag:
        # for each transformed mfov center from section 1 match the reversed_flag transformation
        # matrix (from section 2 to section 1)
        transformed_section_centers2 = [np.dot(m["transformation"]["matrix"],
                                               np.append(mfov_centers2[m["mfov1"]], 1.0))[:2]
                                        for m in pre_mfov_matches["matches"]]
        reversed_transformations = [np.linalg.inv(m["transformation"]["matrix"]) for m in pre_mfov_matches["matches"]]

        # Build a kdtree from the mfovs centers in section 2
        kdtree = spatial.KDTree(np.array(list(mfov_centers1.values())).flatten().reshape(len(mfov_centers1), 2))

        # For each mfov transformed center in section 2, find the closest center, and declare it as a transformation
        closest_centers_idx = kdtree.query(transformed_section_centers2)[1]

        assert (len(reversed_transformations) == len(closest_centers_idx))
        for closest_idx, reversed_transform in zip(closest_centers_idx, reversed_transformations):
            mfov_num2 = sorted_mfovs1[closest_idx]
            transforms[mfov_num2] = reversed_transform

    else:
        for m in pre_mfov_matches["matches"]:
            transforms[m["mfov1"]] = m["transformation"]["matrix"]

    estimated_transforms = {}
    trans_keys = transforms.keys()
    for m in sorted_mfovs1:
        if m not in transforms.keys():
            # Need to find a more intelligent way to do this, but this suffices for now
            # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
            mfov_center = mfov_centers1[m]
            closest_mfov_idx = np.argmin([distance.euclidean(mfov_center, mfov_centers1[mfov]) for mfov in trans_keys])
            estimated_transforms[m] = transforms[list(trans_keys)[closest_mfov_idx]]

    transforms.update(estimated_transforms)

    return transforms


def find_best_mfov_transformation(mfov, best_transformations, mfov_centers):
    """
    Returns a matrix that represents the best transformation for a given mfov to the other section
    :param mfov:
    :param best_transformations:
    :param mfov_centers:
    :return:
    """
    if mfov in best_transformations.keys():
        return best_transformations[mfov]
    # Need to find a more intelligent way to do this, but this suffices for now
    # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
    mfov_center = mfov_centers[mfov]
    trans_keys = best_transformations.keys()
    closest_mfov_idx = np.argmin([distance.euclidean(mfov_center, mfov_centers[mfov_key]) for mfov_key in trans_keys])
    # ** Should we add this transformation? maybe not, because we don't want to get a different result when a few
    # ** missing mfov matches occur and the "best transformation" can change when the centers are updated
    return best_transformations[trans_keys[closest_mfov_idx]]


def get_tile_centers_from_json(tilespecs):
    tiles_centers = []
    for tilespec in tilespecs:
        center_x = (tilespec["bbox"][0] + tilespec["bbox"][1]) / 2.0
        center_y = (tilespec["bbox"][2] + tilespec["bbox"][3]) / 2.0
        tiles_centers.append(np.array([center_x, center_y]))
    return tiles_centers


def get_closest_index_to_point(point, centers_tree):
    _, closest_index = centers_tree.query(point)
    return closest_index


def is_point_in_img(tile_ts, point):
    """Returns True if the given point lies inside the image as denoted by the given tile_tilespec"""
    # TODO - instead of checking inside the bbox, need to check inside the polygon after transformation
    img_bbox = tile_ts["bbox"]

    if img_bbox[0] < point[0] < img_bbox[1] and img_bbox[2] < point[1] < img_bbox[3]:
        return True
    return False


def execute_pmcc_matching(img1_center_point, img1_to_img2_transform, img1_scaled_renderer,
                          img2_scaled_renderer, align_args):
    # Assumes that img1_renderer already has the transformation to img2 applied, and is scaled down,
    # and that img2_renderer is already scaled down,
    # and img1_center_point is w/o the transformation to img2 and w/o the scaling
    # Compute the estimated point on img2 with scaling
    img1_center_point_on_img2 = (np.dot(img1_to_img2_transform[:2, :2], img1_center_point) +
                                 img1_to_img2_transform[:2, 2]) * align_args["block_match"]["scaling"]

    # Fetch the template around img1_point (after transformation)
    template_scaled_side = align_args["block_match"]["template_size"] * align_args["block_match"]["scaling"] / 2
    from_x1, from_y1 = img1_center_point_on_img2 - template_scaled_side
    to_x1, to_y1 = img1_center_point_on_img2 + template_scaled_side
    img1_template, img1_template_start_point = img1_scaled_renderer.crop(from_x1, from_y1, to_x1, to_y1)

    # Fetch a large sub-image around img2_point (using search_window_scaled_size)
    search_window_size = align_args["block_match"]["search_window_size_scale"] * \
                         align_args["block_match"]["template_size"]
    search_window_scaled_side = search_window_size * align_args["block_match"]["scaling"] / 2
    from_x2, from_y2 = img1_center_point_on_img2 - search_window_scaled_side
    to_x2, to_y2 = img1_center_point_on_img2 + search_window_scaled_side
    img2_search_window, img2_search_window_start_point = img2_scaled_renderer.crop(from_x2, from_y2, to_x2, to_y2)

    # execute the PMCC match
    # Do template matching
    result, reason, match_val = PMCC_filter.PMCC_match(img2_search_window, img1_template, align_args)
    if result is not None:
        # Compute the location of the matched point on img2 in non-scaled coordinates
        matched_location_scaled = np.array([reason[1], reason[0]]) + np.array([from_x2, from_y2]) + template_scaled_side
        img2_center_point = matched_location_scaled / align_args["block_match"]["scaling"]
        log_controller.debug("{}: match found: {} and {} (orig assumption: {})".format(
            os.getpid(), img1_center_point, img2_center_point,
            img1_center_point_on_img2 / align_args["block_match"]["scaling"]))
        return img1_center_point, img2_center_point, match_val

    return None


def fetch_and_run(q_jobs, add_result_func, img1_to_img2_transform, img1_scaled_renderer,
                  img2_scaled_renderer, align_args):
    while True:
        job = q_jobs.get(block=True)
        if job is None:
            break
        r = execute_pmcc_matching(job, img1_to_img2_transform,
                                  img1_scaled_renderer, img2_scaled_renderer, align_args)
        if r is not None:
            add_result_func(r)
    log_controller.debug("Process {} is finished".format(os.getpid()))


def match_layers_pmcc_matching(tilespecs_file_path1, tilespecs_file_path2, pre_matches_file_path, output_file_path,
                               targeted_mfov, align_args, processes_num=1):
    log_controller.debug("Block-Matching+PMCC layers: {} with {} targeted mfov: {}".format(tilespecs_file_path1,
                                                                                           tilespecs_file_path2,
                                                                                           targeted_mfov))

    # Parameters for the matching
    log_controller.debug("Actual template size: {} and "
                         "window search size: {} "
                         "(after scaling)".format(align_args["block_match"]["template_size"] *
                                                  align_args["block_match"]["scaling"],
                                                  align_args["block_match"]["search_window_size_scale"] *
                                                  align_args["block_match"]["template_size"] *
                                                  align_args["block_match"]["scaling"]))

    # Read the tilespecs
    tilespecs_file_path1 = os.path.abspath(tilespecs_file_path1)
    tilespecs_file_path2 = os.path.abspath(tilespecs_file_path2)
    ts1 = utils.load_json_file(tilespecs_file_path1)
    ts2 = utils.load_json_file(tilespecs_file_path2)
    indexed_ts1 = utils.index_tilespec(ts1)
    indexed_ts2 = utils.index_tilespec(ts2)

    sorted_mfovs1 = sorted(indexed_ts1.keys())

    # Get the tiles centers for each section
    tile_centers1 = get_tile_centers_from_json(ts1)
    tile_centers1tree = spatial.KDTree(tile_centers1)
    mfov_centers1 = get_mfov_centers_from_json(indexed_ts1)
    mfov_centers2 = get_mfov_centers_from_json(indexed_ts2)

    # Load the preliminary matches
    with open(pre_matches_file_path, 'r') as data_matches:
        mfov_pre_matches = json.load(data_matches)
    if len(mfov_pre_matches["matches"]) == 0:
        log_controller.warning("No matches were found in pre-matching, aborting Block-Matching"
                               " proces between layers: {} and {}".format(tilespecs_file_path1, tilespecs_file_path2))
        return
    best_transformations = get_best_transformations(mfov_pre_matches, tilespecs_file_path1, tilespecs_file_path2,
                                                    mfov_centers1, mfov_centers2, sorted_mfovs1)

    # Create output dictionary
    out_jsonfile = {
        'tilespec1': tilespecs_file_path1,
        'tilespec2': tilespecs_file_path2
    }

    # Create the (lazy) renderers for the two sections
    img1_renderer = TilespecAffineRenderer(ts1)
    img2_renderer = TilespecAffineRenderer(ts2)

    # Generate a hexagonal grid according to the first section's bounding box
    log_controller.debug("Generating Hexagonal Grid")
    bb = BoundingBox.read_bbox(tilespecs_file_path1)
    log_controller.debug("bounding_box: {}".format(bb))
    hex_grid = utils.generate_hexagonal_grid(bb, align_args["block_match"]["hex_spacing"])
    # a single mfov is targeted, so restrict the hexagonal grid to that mfov locations
    mfov_tiles = indexed_ts1[targeted_mfov]
    bb_mfov = BoundingBox.read_bbox_from_ts(mfov_tiles.values())
    log_controller.debug("Trimming bounding box grid points to {} (mfov {})".format(bb_mfov.toArray(), targeted_mfov))
    hex_grid = [p for p in hex_grid if bb_mfov.contains(np.array([p]))]
    log_controller.debug("Found {} possible points in bbox".format(len(hex_grid)))

    # Use the mfov expected transform (from section 1 to section 2) to transform img1
    img1_to_img2_transform = np.array(find_best_mfov_transformation(targeted_mfov,
                                                                    best_transformations,
                                                                    mfov_centers1)[:2])
    img1_renderer.add_transformation(img1_to_img2_transform)

    # Scale down the rendered images
    scale_transformation = np.array([
        [align_args["block_match"]["scaling"], 0., 0.],
        [0., align_args["block_match"]["scaling"], 0.]
    ])
    img1_renderer.add_transformation(scale_transformation)
    img2_renderer.add_transformation(scale_transformation)

    # Execute PMCC Matching
    log_controller.debug("Performing PMCC Matching with {} processes".format(processes_num))
    # Allocate processes_num-1 other processes and initialize with the "static" data,
    # and a queue for jobs and a queue for results
    q_jobs = mp.Queue(maxsize=len(hex_grid))
    # Creating the results queue using a manager, or otherwise we enter a deadlock because buffers aren't flushed
    mp_manager = mp.Manager()
    q_res = mp_manager.Queue(maxsize=len(hex_grid))

    all_processes = [mp.Process(target=fetch_and_run,
                                args=(q_jobs, lambda x: q_res.put(x), img1_to_img2_transform,
                                      img1_renderer, img2_renderer, align_args)) for i in range(processes_num - 1)]
    for p in all_processes:
        p.start()

    # Iterate over the hexagonal grid points, and only check those that are part of the targeted mfov
    on_section_points_num = 0
    for i in range(len(hex_grid)):
        # Find the tile image where the point from the hexagonal is in the first section
        img1_ind = get_closest_index_to_point(hex_grid[i], tile_centers1tree)
        if img1_ind is None:
            continue
        if ts1[img1_ind]["mfov"] != targeted_mfov:
            continue
        if not is_point_in_img(ts1[img1_ind], hex_grid[i]):
            continue

        img1_point = np.array(hex_grid[i])
        on_section_points_num += 1

        # Perform matching of that point
        q_jobs.put(img1_point)

    # Add empty jobs to end the execution of each process
    for i in range(processes_num):
        q_jobs.put(None)

    # Used to store the results of the main process (and then all the results)
    point_matches = []

    # Use the main process to run jobs like any other process
    fetch_and_run(q_jobs, lambda x: point_matches.append(x), img1_to_img2_transform,
                  img1_renderer, img2_renderer, align_args)

    # Wait for the termination of the other processes
    log_controller.debug("Waiting for other processes to finish")
    for p in all_processes:
        p.join()

    log_controller.debug("Collecting results")
    while not q_res.empty():
        r = q_res.get(block=True)
        point_matches.append(r)

    log_controller.debug("Found {} matches out of possible {} "
                         "points (on section points: {})".format(len(point_matches),
                                                                 len(hex_grid),
                                                                 on_section_points_num))

    # Save the output
    log_controller.debug("Saving output to: {}".format(output_file_path))
    out_jsonfile['mesh'] = hex_grid
    if targeted_mfov != -1:
        out_jsonfile['mfov1'] = targeted_mfov

    final_point_matches = []
    for pm in point_matches:
        p1, p2, match_val = pm
        record = {
            'point1': p1.tolist(),
            'point2': p2.tolist(),
            'match_val': float(match_val)
        }
        final_point_matches.append(record)

    out_jsonfile['pointmatches'] = final_point_matches
    with open(output_file_path, 'w') as out:
        json.dump(out_jsonfile, out, indent=4)

    log_controller.debug("Done")
