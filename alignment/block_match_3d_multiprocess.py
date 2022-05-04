# Setup
from __future__ import print_function
import os
import numpy as np
import json
import time
import sys
from scipy.spatial import distance
from scipy import spatial
import cv2
import argparse
from ..common import utils
from ..common.bounding_box import BoundingBox
import PMCC_filter
import multiprocessing as mp
#import pyximport
#pyximport.install()
from rh_renderer.tilespec_affine_renderer import TilespecAffineRenderer
import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger_handler = logging.StreamHandler()
logger_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)



def get_mfov_centers_from_json(indexed_ts):
    mfov_centers = {}
    for mfov in indexed_ts.keys():
        tile_bboxes = []
        mfov_tiles = indexed_ts[mfov].values()
        tile_bboxes = zip(*[tile["bbox"] for tile in mfov_tiles])
        min_x = min(tile_bboxes[0])
        max_x = max(tile_bboxes[1])
        min_y = min(tile_bboxes[2])
        max_y = max(tile_bboxes[3])
        # center = [(min_x + min_y) / 2.0, (min_y + max_y) / 2.0], but w/o overflow
        mfov_centers[mfov] = np.array([(min_x / 2.0 + max_x / 2.0, min_y / 2.0 + max_y / 2.0)])
    return mfov_centers

def get_best_transformations(pre_mfov_matches, tiles_fname1, tiles_fname2, mfov_centers1, mfov_centers2, sorted_mfovs1, sorted_mfovs2):
    """Returns a dictionary that maps an mfov number to a matrix that best describes the transformation to the other section.
       As not all mfov's may be matched, some mfovs will be missing from the dictionary.
       If the given tiles file names are reversed, an inverted matrix is returned."""
    transforms = {}
    reversed = False
    if tiles_fname1 == pre_mfov_matches["tilespec1"] and \
       tiles_fname2 == pre_mfov_matches["tilespec2"]:
        reversed = False
    elif tiles_fname1 == pre_mfov_matches["tilespec2"] and \
         tiles_fname2 == pre_mfov_matches["tilespec1"]:
        reversed = True
    else:
        logger.error("Error: could not find pre_matches between tilespecs {} and {} (found tilespecs {} and {} instead!).".format(tiles_fname1, tiles_fname2, pre_mfov_matches["tilespec1"], pre_mfov_matches["tilespec2"]))
        return {}

    if reversed:
        # for each transformed mfov center from section 1 match the reversed transformation matrix (from section 2 to section 1)
        transformed_section_centers2 = [np.dot(m["transformation"]["matrix"], np.append(mfov_centers2[m["mfov1"]], 1.0))[:2] for m in pre_mfov_matches["matches"]]
        reversed_transformations = [np.linalg.inv(m["transformation"]["matrix"]) for m in pre_mfov_matches["matches"]]

        # Build a kdtree from the mfovs centers in section 2
        kdtree = spatial.KDTree(np.array(mfov_centers1.values()).flatten().reshape(len(mfov_centers1), 2))

        # For each mfov transformed center in section 2, find the closest center, and declare it as a transformation
        closest_centers_idx = kdtree.query(transformed_section_centers2)[1]
        
        assert(len(reversed_transformations) == len(closest_centers_idx))
        for closest_idx, reversed_transform in zip(closest_centers_idx, reversed_transformations):
            mfov_num2 = sorted_mfovs1[closest_idx]
            transforms[mfov_num2] = reversed_transform

    else:
        for m in pre_mfov_matches["matches"]:
            transforms[m["mfov1"]] = m["transformation"]["matrix"]

#    for m in pre_mfov_matches["matches"]:
#        if reversed:
#            cur_matrix = m["transformation"]["matrix"]
#            section_center1 = mfov_centers1[m["mfov1"] - 1]
#            transformed_section_center2 = np.dot(cur_matrix, section_center1)
#            # Reverse the matrix
#            rev_matrix = np.linalg.inv(cur_matrix)
#
#            # Find the mfov that is closest to the one that is being transformed
#            closest_mfov_num2 = np.argmin([((c[0] - transformed_section_center2[0])**2 + (c[1] - transformed_section_center2[1])**2) for c in mfov_centers2]) + 1
#
#            transforms[m["mfov2"]] = rev_matrix
#        else:
#            transforms[m["mfov1"]] = m["transformation"]["matrix"]

    # Add the "transformations" of all the mfovs w/o a direct mapping (using their neighbors)
    estimated_transforms = {}
    trans_keys = transforms.keys()
    for m in sorted_mfovs1:
        if m not in transforms.keys():
            # Need to find a more intelligent way to do this, but this suffices for now
            # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
            mfov_center = mfov_centers1[m]
            closest_mfov_idx = np.argmin([distance.euclidean(mfov_center, mfov_centers1[mfovk]) for mfovk in trans_keys])
            estimated_transforms[m] = transforms[trans_keys[closest_mfov_idx]]
 
    transforms.update(estimated_transforms)
           
    return transforms


def find_best_mfov_transformation(mfov, best_transformations, mfov_centers):
    """Returns a matrix that represnets the best transformation for a given mfov to the other section"""
    if mfov in best_transformations.keys():
        return best_transformations[mfov]
    # Need to find a more intelligent way to do this, but this suffices for now
    # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
    mfov_center = mfov_centers[mfov]
    trans_keys = best_transformations.keys()
    closest_mfov_idx = np.argmin([distance.euclidean(mfov_center, mfov_centers[mfovk]) for mfovk in trans_keys])
    # ** Should we add this transformation? maybe not, because we don't want to get a different result when a few
    # ** missing mfov matches occur and the "best transformation" can change when the centers are updated
    return best_transformations[trans_keys[closest_mfov_idx]]


def get_tile_centers_from_json(ts):
    tiles_centers = []
    for tile in ts:
        center_x = (tile["bbox"][0] + tile["bbox"][1]) / 2.0
        center_y = (tile["bbox"][2] + tile["bbox"][3]) / 2.0
        tiles_centers.append(np.array([center_x, center_y]))
    return tiles_centers


def get_closest_index_to_point(point, centerstree):
    distanc, closest_index = centerstree.query(point)
    return closest_index

def is_point_in_img(tile_ts, point):
    """Returns True if the given point lies inside the image as denoted by the given tile_tilespec"""
    # TODO - instead of checking inside the bbox, need to check inside the polygon after transformation
    img_bbox = tile_ts["bbox"]

    if point[0] > img_bbox[0] and point[1] > img_bbox[2] and \
       point[0] < img_bbox[1] and point[1] < img_bbox[3]:
        return True
    return False


def execute_pmcc_matching(img1_center_point, img1_to_img2_transform, scaling, template_size, search_window_size, img1_scaled_renderer, img2_scaled_renderer, min_corr, max_curvature, max_rod, debug_save_matches=False, debug_dir=None):
    # Assumes that img1_renderer already has the transformation to img2 applied, and is scaled down,
    # and that img2_renderer is already scaled down,
    # and img1_center_point is w/o the transformation to img2 and w/o the scaling

    # Compute the estimated point on img2 with scaling
    img1_center_point_on_img2 = (np.dot(img1_to_img2_transform[:2,:2], img1_center_point) + img1_to_img2_transform[:2,2]) * scaling

    # Fetch the template around img1_point (after transformation)
    template_scaled_side = template_size * scaling / 2
    from_x1, from_y1 = img1_center_point_on_img2 - template_scaled_side
    to_x1, to_y1 = img1_center_point_on_img2 + template_scaled_side
    img1_template, img1_template_start_point = img1_scaled_renderer.crop(from_x1, from_y1, to_x1, to_y1)
    
    # Fetch a large sub-image around img2_point (using search_window_scaled_size)
    search_window_scaled_side = search_window_size * scaling / 2
    from_x2, from_y2 = img1_center_point_on_img2 - search_window_scaled_side
    to_x2, to_y2 = img1_center_point_on_img2 + search_window_scaled_side
    img2_search_window, img2_search_window_start_point = img2_scaled_renderer.crop(from_x2, from_y2, to_x2, to_y2)

    # execute the PMCC match
    # Do template matching
    result, reason, match_val = PMCC_filter.PMCC_match(img2_search_window, img1_template, min_correlation=min_corr, maximal_curvature_ratio=max_curvature, maximal_ROD=max_rod)
    if result is not None:
        # Compute the location of the matched point on img2 in non-scaled coordinates
        #matched_location_scaled = np.array([reason[1], reason[0]]) + template_scaled_side
        #img2_center_point = (matched_location_scaled + img1_center_point_on_img2) / scaling 
        matched_location_scaled = np.array([reason[1], reason[0]]) + np.array([from_x2, from_y2]) + template_scaled_side
        img2_center_point = matched_location_scaled / scaling 
        logger.debug("{}: match found: {} and {} (orig assumption: {})".format(os.getpid(), img1_center_point, img2_center_point, img1_center_point_on_img2 / scaling))
        if debug_save_matches:
            #debug_out_fname1 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image1.png".format(hexgr_point[0], hexgr_point[1], reasonx, reasony))
            #debug_out_fname2 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image2.png".format(hexgr_point[0], hexgr_point[1], reasonx, reasony))
            debug_out_fname1 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image1.png".format(int(img1_center_point[0]), int(img1_center_point[1]), int(img2_center_point[0]), int(img2_center_point[1])))
            debug_out_fname2 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image2.png".format(int(img1_center_point[0]), int(img1_center_point[1]), int(img2_center_point[0]), int(img2_center_point[1])))
            cv2.imwrite(debug_out_fname1, img1_template)
            img2_cut_out = img2_search_window[reason[0]:reason[0] + 2 * template_scaled_side, reason[1]:reason[1] + 2 * template_scaled_side]
            cv2.imwrite(debug_out_fname2, img2_cut_out)
        return img1_center_point, img2_center_point, match_val

    # When there are no matches save template and search window
#    if debug_save_matches:
#        debug_out_fname1 = os.path.join(debug_dir, "debug_no_match_sec1{}-{}_sec2{}-{}_image1.png".format(int(img1_center_point[0]), int(img1_center_point[1]), int(img1_center_point_on_img2[0] / scaling), int(img1_center_point_on_img2[1] / scaling)))
#        debug_out_fname2 = os.path.join(debug_dir, "debug_no_match_sec1{}-{}_sec2{}-{}_image2_window.png".format(int(img1_center_point[0]), int(img1_center_point[1]), int(img1_center_point_on_img2[0] / scaling), int(img1_center_point_on_img2[1] / scaling)))
#        cv2.imwrite(debug_out_fname1, img1_template)
#        cv2.imwrite(debug_out_fname2, img2_search_window)

    #return img1_center_point, None, None
    return None


def fetch_and_run(q_jobs, add_result_func, img1_to_img2_transform, scaling, template_size, search_window_size, img1_scaled_renderer, img2_scaled_renderer, min_corr, max_curvature, max_rod, debug_save_matches=False, deubg_dir=None):
    while True:
        job = q_jobs.get(block=True)
        if job is None:
            break
        #logger.debug("Working on point: {}".format(job))
        r = execute_pmcc_matching(job, img1_to_img2_transform, scaling, template_size, search_window_size, img1_scaled_renderer, img2_scaled_renderer, min_corr, max_curvature, max_rod, debug_save_matches, deubg_dir)
        if r is not None:
            add_result_func(r)
    logger.info("Process {} is finished".format(os.getpid()))

def match_layers_pmcc_matching(tiles_fname1, tiles_fname2, pre_matches_fname, out_fname, targeted_mfov, conf_fname=None, processes_num=1):
    starttime = time.time()
    logger.info("Block-Matching+PMCC layers: {} with {} targeted mfov: {}".format(tiles_fname1, tiles_fname2, targeted_mfov))

    # Load parameters file
    params = utils.conf_from_file(conf_fname, 'MatchLayersBlockMatching')
    if params is None:
        params = {}

    # Parameters for the matching
    hex_spacing = params.get("hex_spacing", 1500)
    scaling = params.get("scaling", 0.2)
    template_size = params.get("template_size", 200)
    search_window_size = params.get("search_window_size", 8 * template_size)
    logger.info("Actual template size: {} and window search size: {} (after scaling)".format(template_size * scaling, search_window_size * scaling))

    # Parameters for PMCC filtering
    min_corr = params.get("min_correlation", 0.2)
    max_curvature = params.get("maximal_curvature_ratio", 10)
    max_rod = params.get("maximal_ROD", 0.9)

    debug_save_matches = False
    debug_dir = None
    if "debug_save_matches" in params.keys():
        logger.info("Debug mode - on")
        debug_save_matches = True
        if debug_save_matches:
            # Create a debug directory
            import datetime
            debug_dir = os.path.join(os.path.dirname(out_fname), 'debug_matches_{}'.format(datetime.datetime.now().isoformat()))
            os.mkdir(debug_dir)

    # Read the tilespecs
    tiles_fname1 = os.path.abspath(tiles_fname1)
    tiles_fname2 = os.path.abspath(tiles_fname2)
    ts1 = utils.load_tilespecs(tiles_fname1)
    ts2 = utils.load_tilespecs(tiles_fname2)
    indexed_ts1 = utils.index_tilespec(ts1)
    indexed_ts2 = utils.index_tilespec(ts2)

    sorted_mfovs1 = sorted(indexed_ts1.keys())
    sorted_mfovs2 = sorted(indexed_ts2.keys())

    # Get the tiles centers for each section
    tile_centers1 = get_tile_centers_from_json(ts1)
    tile_centers1tree = spatial.KDTree(tile_centers1)
    tile_centers2 = get_tile_centers_from_json(ts2)
    tile_centers2tree = spatial.KDTree(tile_centers2)
    mfov_centers1 = get_mfov_centers_from_json(indexed_ts1)
    mfov_centers2 = get_mfov_centers_from_json(indexed_ts2)

    # Load the preliminary matches
    with open(pre_matches_fname, 'r') as data_matches:
        mfov_pre_matches = json.load(data_matches)
    if len(mfov_pre_matches["matches"]) == 0:
        logger.warn("No matches were found in pre-matching, aborting Block-Matching proces between layers: {} and {}".format(tiles_fname1, tiles_fname2))
        return
    best_transformations = get_best_transformations(mfov_pre_matches, tiles_fname1, tiles_fname2, mfov_centers1, mfov_centers2, sorted_mfovs1, sorted_mfovs2)

    # Create output dictionary
    out_jsonfile = {}
    out_jsonfile['tilespec1'] = tiles_fname1
    out_jsonfile['tilespec2'] = tiles_fname2

    # Create the (lazy) renderers for the two sections
    img1_renderer = TilespecAffineRenderer(ts1)
    img2_renderer = TilespecAffineRenderer(ts2)

    # Generate an hexagonal grid according to the first section's bounding box
    logger.info("Generating Hexagonal Grid")
    bb = BoundingBox.read_bbox(tiles_fname1)
    logger.info("bounding_box: {}".format(bb))
    hexgr = utils.generate_hexagonal_grid(bb, hex_spacing)
    #print(hexgr)
    # a single mfov is targeted, so restrict the hexagonal grid to that mfov locations
    mfov_tiles = indexed_ts1[targeted_mfov]
    bb_mfov = BoundingBox.read_bbox_from_ts(mfov_tiles.values())
    logger.info("Trimming bounding box grid points to {} (mfov {})".format(bb_mfov.toArray(), targeted_mfov))
    hexgr = [p for p in hexgr if bb_mfov.contains(np.array([p]))]
    logger.info("Found {} possible points in bbox".format(len(hexgr)))

    # Use the mfov exepected transform (from section 1 to section 2) to transform img1
    img1_to_img2_transform = np.array(find_best_mfov_transformation(targeted_mfov, best_transformations, mfov_centers1)[:2])
    img1_renderer.add_transformation(img1_to_img2_transform)

    # Scale down the rendered images
    scale_transformation = np.array([
                                [ scaling, 0., 0. ],
                                [ 0., scaling, 0. ]
                            ])
    img1_renderer.add_transformation(scale_transformation)
    img2_renderer.add_transformation(scale_transformation)

    # Execute PMCC Matching
    logger.info("Performing PMCC Matching with {} processes".format(processes_num))
    # Allocate processes_num-1 other processes and initialize with the "static" data, and a queue for jobs and a queue for results
    q_jobs = mp.Queue(maxsize=len(hexgr))
    # Creating the results queue using a manager, or otherwise we enter a deadlock because buffers aren't flushed
    mp_manager = mp.Manager()
    q_res = mp_manager.Queue(maxsize=len(hexgr))

    all_processes = [mp.Process(target=fetch_and_run, args=(q_jobs, lambda x: q_res.put(x), img1_to_img2_transform, scaling, template_size, search_window_size, img1_renderer, img2_renderer, min_corr, max_curvature, max_rod, debug_save_matches, debug_dir)) for i in range(processes_num - 1)]
    for p in all_processes:
        p.start()

    # Iterate over the hexagonal grid points, and only check those that are part of the targeted mfov
    on_section_points_num = 0
    for i in range(len(hexgr)):
        # Find the tile image where the point from the hexagonal is in the first section
        img1_ind = get_closest_index_to_point(hexgr[i], tile_centers1tree)
        if img1_ind is None:
            continue
        if ts1[img1_ind]["mfov"] != targeted_mfov:
            continue
        if not is_point_in_img(ts1[img1_ind], hexgr[i]):
            continue

        img1_point = np.array(hexgr[i])
        on_section_points_num += 1
        
        # Perform matching of that point
        q_jobs.put(img1_point)
 
    # Add empty jobs to end the execution of each process
    for i in range(processes_num):
        q_jobs.put(None)

    # Used to store the results of the main process (and then all the results)
    point_matches = []

    # Use the main process to run jobs like any other process
    fetch_and_run(q_jobs, lambda x: point_matches.append(x), img1_to_img2_transform, scaling, template_size, search_window_size, img1_renderer, img2_renderer, min_corr, max_curvature, max_rod, debug_save_matches, debug_dir)

    # Wait for the termination of all other processes
    logger.info("Waiting for other processes to finish")
    for p in all_processes:
        p.join()

    logger.info("Collecting results")
    while not q_res.empty():
        r = q_res.get(block=True)
        point_matches.append(r)

    logger.info("Found {} matches out of possible {} points (on section points: {})".format(len(point_matches), len(hexgr), on_section_points_num))


    # Save the output
    logger.info("Saving output to: {}".format(out_fname))
    out_jsonfile['runtime'] = time.time() - starttime
    out_jsonfile['mesh'] = hexgr
    if targeted_mfov != -1:
        out_jsonfile['mfov1'] = targeted_mfov

    final_point_matches = []
    for pm in point_matches:
        p1, p2, match_val = pm
        record = {}
        record['point1'] = p1.tolist()
        record['point2'] = p2.tolist()
        #record['isvirtualpoint'] = nmesh
        record['match_val'] = float(match_val)
        final_point_matches.append(record)

    out_jsonfile['pointmatches'] = final_point_matches
    with open(out_fname, 'w') as out:
        json.dump(out_jsonfile, out, indent=4)

    logger.info("Done")


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given two tilespecs of two sections, and a preliminary matches list, generates a grid the image, and performs block matching (with PMCC filtering).')
    parser.add_argument('tiles_file1', metavar='tiles_file1', type=str,
                        help='the first layer json file containing tilespecs')
    parser.add_argument('tiles_file2', metavar='tiles_file2', type=str,
                        help='the second layer json file containing tilespecs')
    parser.add_argument('pre_matches_file', metavar='pre_matches_file', type=str,
                        help='a json file that contains the preliminary matches')
    parser.add_argument('mfov', type=int,
                        help='the mfov number of compare')
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output correspondent_spec file, that will include the matches between the sections (default: ./matches.json)',
                        default='./matches.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads (processes) to use (default: 1)',
                        default=1)

    args = parser.parse_args()
    match_layers_pmcc_matching(args.tiles_file1, args.tiles_file2,
                               args.pre_matches_file, args.output_file,
                               args.mfov,
                               conf_fname=args.conf_file_name, processes_num=args.threads_num)

if __name__ == '__main__':
    main()
