import os
import numpy as np
import h5py
import json
import cv2
import common.trans_models as models
from common import ransac
from common import utils
from common.bounding_box import BoundingBox

overall_args = utils.load_json_file('arguments/overall_args.json')
log_controller = utils.LogController('alignment', 'pre_match', os.path.join(overall_args["base"]["workspace"], 'log'))
if overall_args["base"]["EM_type"] == 'singlebeam':
    TILES_PER_MFOV = 1
else:
    TILES_PER_MFOV = 61


def load_features(h5_file_path: str, tilespec: dict):
    """
    Load the features from h5py file, and apply the new transformation model on the points. (Generated from stitching)
    :param h5_file_path:
    :param tilespec:
    :return:
    """
    log_controller.debug('Loading features from file: {}'.format(h5_file_path))
    # Should have the same name as the following: [tilespec base filename]_[img filename].json/.hdf5
    if not os.path.basename(os.path.splitext(tilespec["mipmapLevels"]["0"]["imageUrl"])[0]) in h5_file_path:
        log_controller.error('Features file does not match tilespec')
        raise AssertionError

    with h5py.File(h5_file_path, 'r') as f:
        descs = f['descs'][:]
        octaves = f['pts']['octaves'][:]
        locations = np.array(f['pts']['locations'])

    # If no relevant features are found, return an empty set
    if len(locations) == 0:
        return np.array([]).reshape((0, 2)), [], []

    cur_octave = (octaves.astype(int) & 0xff)
    cur_octave[cur_octave > 127] -= 255
    mask = (cur_octave == 4) | (cur_octave == 5)
    points = locations[mask, :]
    descs = descs[mask]

    # Apply the transformation to each point
    new_model = models.Transforms.from_tilespec(tilespec["transforms"][0])
    points = new_model.apply(points)

    return points, descs


def get_center(mfov_tilespecs: dict):
    """
    Calculate the center location of the mfov
    :param mfov_tilespecs:
    :return:
    """
    x_loc_sum, y_loc_sum, num_pts = 0, 0, 0
    for tilespec in mfov_tilespecs.values():
        x_loc_sum += tilespec["bbox"][0] + tilespec["bbox"][1]
        y_loc_sum += tilespec["bbox"][2] + tilespec["bbox"][3]
        num_pts += 2
    return [x_loc_sum / num_pts, y_loc_sum / num_pts]


def analyze_mfov(mfov_tilespecs: dict, features_folder_path: str):
    """
    Analyze mfov
    :param mfov_tilespecs:
    :param features_folder_path:
    :return:
    """
    """Returns all the relevant features of the tiles in a single mfov"""
    all_pts = np.array([]).reshape((0, 2))
    all_descs = []

    mfov_num = int(list(mfov_tilespecs.values())[0]["mfov"])
    mfov_string = ("%06d" % mfov_num)
    mfov_feature_file_paths = utils.ls_absolute_paths(os.path.join(features_folder_path, mfov_string))
    if len(mfov_feature_file_paths) < TILES_PER_MFOV:
        log_controller.warning("The number of feature files in directory: {} is smaller than {}".format(
            os.path.join(os.path.join(features_folder_path, mfov_string)), TILES_PER_MFOV))

    # load each features file, and concatenate all to single lists
    for tile_num in mfov_tilespecs.keys():
        # for feature_file in mfov_feature_file_paths:
        feature_file = [file_path for file_path in mfov_feature_file_paths if
                        "_{}_{}_".format(mfov_string, "%03d" % tile_num) in file_path.split('sifts_')[1]][0]
        # Get the correct tile tilespec from the section tilespec (convert to int to remove leading zeros)
        # tile_num = int(feature_file.split('sifts_')[1].split('_')[2])
        (tmp_pts, tmp_descs) = load_features(feature_file, mfov_tilespecs[tile_num])
        if type(tmp_descs) is not list:
            # concatenate the results
            all_pts = np.append(all_pts, tmp_pts, axis=0)
            all_descs.append(tmp_descs)
    all_pts = np.array(all_pts)
    return all_pts, np.vstack(all_descs)


def generate_matches_crosscheck_cv2(all_pts1: np.array, all_pts2: np.array, all_descs1: list, all_descs2: list):
    """
    Crosscheck match
    :param all_pts1:
    :param all_pts2:
    :param all_descs1:
    :param all_descs2:
    :return:
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.knnMatch(np.array(all_descs1), np.array(all_descs2), k=1)
    good_matches = [m for m in matches if len(m) > 0]
    match_points = np.array([
        np.array([all_pts1[[m[0].queryIdx for m in good_matches]]][0]),
        np.array([all_pts2[[m[0].trainIdx for m in good_matches]]][0])])
    return match_points


def compare_features(section1_pts_descs: list, section2_pts_descs: list, align_args: dict):
    """
    :param section1_pts_descs:
    :param section2_pts_descs:
    :param align_args:
    :return:
    """
    [all_pts1, all_descs1] = section1_pts_descs
    [all_pts2, all_descs2] = section2_pts_descs
    log_controller.debug("len(all_pts1): {}, all_descs1.shape: {}".format(len(all_pts1), all_descs1.shape))
    log_controller.debug("len(all_pts2): {}, all_descs2.shape: {}".format(len(all_pts2), all_descs2.shape))
    match_points = generate_matches_crosscheck_cv2(all_pts1, all_pts2, all_descs1, all_descs2)

    if match_points.shape[0] == 0 or match_points.shape[1] == 0:
        return None, 0, 0, 0, len(all_pts1), len(all_pts2)

    model_index = align_args["pre_match"]["model_index"]
    iterations = align_args["pre_match"]["iterations"]
    max_epsilon = align_args["pre_match"]["max_epsilon"]
    min_inlier_ratio = align_args["pre_match"]["min_inlier_ratio"]
    min_num_inlier = align_args["pre_match"]["min_num_inlier"]
    max_trust = align_args["pre_match"]["max_trust"]
    det_delta = align_args["pre_match"]["det_delta"]
    max_stretch = align_args["pre_match"]["max_stretch"]
    model, filtered_matches = ransac.filter_matches(match_points, model_index, iterations, max_epsilon,
                                                    min_inlier_ratio, min_num_inlier, max_trust, det_delta, max_stretch)
    if filtered_matches is None:
        filtered_matches = np.zeros((0, 0))
    return (model, filtered_matches.shape[1],
            float(filtered_matches.shape[1]) / match_points.shape[1], match_points.shape[1],
            len(all_pts1), len(all_pts2))


def load_mfovs_features(indexed_ts: dict, features_dir: str, mfovs_idx: list):
    """
    :param indexed_ts:
    :param features_dir:
    :param mfovs_idx:
    :return:
    """
    all_pts, all_descs = np.array([]).reshape((0, 2)), []
    for idx in mfovs_idx:
        mfov_points, mfov_descs = analyze_mfov(indexed_ts[idx], features_dir)
        all_pts = np.append(all_pts, mfov_points, axis=0)
        all_descs.append(mfov_descs)
    return all_pts, all_descs


def iterative_search(align_args: dict, layer1: int, layer2: int, indexed_ts1: dict, indexed_ts2: dict,
                     features_dir1: str, features_dir2: str, mfovs_nums1: list, centers_mfovs_nums2: list,
                     section2_mfov_bboxes: dict, sorted_mfovs2: list, is_initial_search=False):
    """
    :param align_args:
    :param layer1:
    :param layer2:
    :param indexed_ts1:
    :param indexed_ts2:
    :param features_dir1:
    :param features_dir2:
    :param mfovs_nums1:
    :param centers_mfovs_nums2:
    :param section2_mfov_bboxes:
    :param sorted_mfovs2:
    :param is_initial_search:
    :return:
    """
    # Load the features from the mfovs in section 1
    all_points1, all_descs1 = load_mfovs_features(indexed_ts1, features_dir1, mfovs_nums1)
    section1_pts_descs = [all_points1, np.vstack(all_descs1)]
    log_controller.debug("Section {} - mfovs: {}, {} features loaded.".format(layer1, mfovs_nums1, len(all_points1)))

    # Make sure we have enough features from section 1
    if len(all_points1) < align_args["pre_match"]["min_features_num"]:
        log_controller.debug("Number of features in Section {} "
                             "mfov(s) {} is {}, and smaller than {}. "
                             "Skipping feature matching".format(layer1,
                                                                mfovs_nums1,
                                                                len(all_points1),
                                                                align_args["pre_match"]["min_features_num"]))
        return None, 0, 0, 0, 0, 0, 0

    # Take the mfovs in the middle of the 2nd section as the initial matched area
    # (on each iteration, increase the matched area, by taking all the mfovs that overlap
    # with the bounding box of the previous matched area)
    current_area = BoundingBox.fromList(section2_mfov_bboxes[centers_mfovs_nums2[0]].toArray())
    log_controller.debug("Adding area 0: {}".format(section2_mfov_bboxes[centers_mfovs_nums2[0]].toArray()))
    for i in range(1, len(centers_mfovs_nums2)):
        center_mfov_num2 = centers_mfovs_nums2[i]
        current_area.extend(BoundingBox.fromList(section2_mfov_bboxes[center_mfov_num2].toArray()))
        log_controller.debug("Adding area {}: {}".format(i, section2_mfov_bboxes[center_mfov_num2].toArray()))
    current_mfovs = set(centers_mfovs_nums2)
    current_features_pts, current_features_descs = np.array([]).reshape((0, 2)), []
    for center_mfov_num2 in centers_mfovs_nums2:
        log_controller.debug("loading features for mfov: {}".format(center_mfov_num2))
        mfov_points, mfov_descs = analyze_mfov(indexed_ts2[center_mfov_num2], features_dir2)
        current_features_pts = np.append(current_features_pts, mfov_points, axis=0)
        current_features_descs.append(mfov_descs)
    log_controller.debug("Features loaded")
    current_features = [current_features_pts, np.vstack(current_features_descs)]

    match_found = False
    match_iteration = 0
    best_transform = None
    num_filtered = 0
    filter_rate = 0
    num_rod = 0
    num_m1 = 0
    num_m2 = 0
    # Save the best model that we find through the iterations
    saved_model = {
        'model': None,
        'num_filtered': 0,
        'filter_rate': 0,
        'num_rod': 0,
        'num_m1': 0,
        'num_m2': 0
    }
    while not match_found:
        match_iteration += 1
        log_controller.debug("Iteration {}: using {} mfovs "
                             "from section {} ({} features)".format(match_iteration, len(current_mfovs),
                                                                    layer2, len(current_features_pts)))
        # Try to match the 3-mfovs features of section1 to the current features of section2
        (model, num_filtered, filter_rate, num_rod, num_m1, num_m2) = compare_features(section1_pts_descs,
                                                                                       current_features, align_args)

        if model is None:
            if saved_model['model'] is not None:
                log_controller.debug("Could not find a valid model between Sec{} "
                                     "and Sec{} in iteration {}, but found one before, "
                                     "so stopping search".format(layer1, layer2, match_iteration))
                break
            log_controller.debug("Could not find a valid model between "
                                 "Sec{} and Sec{} in iteration {}".format(layer1, layer2, match_iteration))
        else:
            log_controller.debug("Found a model {} (with {} matches) between Sec{} and Sec{} in "
                                 "iteration {}, need to verify cutoff".format(model.to_str(), num_filtered,
                                                                              layer1, layer2, match_iteration))

            if num_filtered >= saved_model['num_filtered']:
                saved_model['model'] = model
                saved_model['num_filtered'] = num_filtered
                saved_model['filter_rate'] = filter_rate
                saved_model['num_rod'] = num_rod
                saved_model['num_m1'] = num_m1
                saved_model['num_m2'] = num_m2
            elif num_filtered < saved_model['num_filtered']:
                # if already passed the highest number of matches that can be found
                # just use the best that was found, and no need to search more
                break

        if num_filtered > (
                align_args["pre_match"]["num_filtered_percent"] * len(all_points1) / len(mfovs_nums1)) and \
                filter_rate > align_args["pre_match"]["filter_rate_cutoff"]:
            best_transform = model
            match_found = True
        else:
            # Find the mfovs that are overlapping with the current area
            filter_thres = align_args["pre_match"]["num_filtered_percent"] * len(all_points1) / len(mfovs_nums1)
            log_controller.debug("len(mfovs_nums1):{}".format(len(mfovs_nums1)))
            log_controller.debug("threshold wasn't met: num_filtered: {} > {} and "
                                 "filter_rate: {} > {}".format(num_filtered, filter_thres,
                                                               filter_rate,
                                                               align_args["pre_match"]["filter_rate_cutoff"]))
            overlapping_mfovs = set()
            for m in sorted_mfovs2:
                if current_area.overlap(section2_mfov_bboxes[m]):
                    overlapping_mfovs.add(m)

            new_mfovs = overlapping_mfovs - current_mfovs
            if len(new_mfovs) == 0:
                # No new mfovs were found, giving up
                log_controller.debug("No model found between sections {} and {}, "
                                     "and no more mfovs were found. Giving up!".format(layer1, layer2))
                break

            # Add the new mfovs features
            log_controller.debug("Adding {} mfovs ({}) to the second layer".format(len(new_mfovs), new_mfovs))
            for m in new_mfovs:
                mfov_points, mfov_descs = analyze_mfov(indexed_ts2[m], features_dir2)
                current_features_pts = np.append(current_features_pts, mfov_points, axis=0)
                current_features_descs.append(mfov_descs)

                # Expand the current area
                current_area.extend(section2_mfov_bboxes[m])
            log_controller.debug("Combining features")
            current_features = (current_features_pts, np.vstack(current_features_descs))
            current_mfovs = overlapping_mfovs

        if not is_initial_search and match_iteration == align_args["pre_match"]["max_attempts"]:
            log_controller.debug("Reached maximal number of attempts in iterative search, stopping search")

    if best_transform is None and saved_model['model'] is not None:
        best_transform = saved_model['model']
        num_filtered = saved_model['num_filtered']
        filter_rate = saved_model['filter_rate']
        num_rod = saved_model['num_rod']
        num_m1 = saved_model['num_m1']
        num_m2 = saved_model['num_m2']

    return best_transform, num_filtered, filter_rate, num_rod, num_m1, num_m2, match_iteration


def analyze_slices(tilespecs_file_path1: str, tilespecs_file_path2: str,
                   features_folder_path1: str, features_folder_path2: str, align_args: dict):
    """
    Analyze slices
    :param tilespecs_file_path1:
    :param tilespecs_file_path2:
    :param features_folder_path1:
    :param features_folder_path2:
    :param align_args:
    :return:
    """
    ts1 = utils.load_json_file(tilespecs_file_path1)
    ts2 = utils.load_json_file(tilespecs_file_path2)
    indexed_ts1 = utils.index_tilespec(ts1)
    indexed_ts2 = utils.index_tilespec(ts2)

    num_mfovs1 = len(indexed_ts1)
    num_mfovs2 = len(indexed_ts2)

    sorted_mfovs1 = sorted(indexed_ts1.keys())
    sorted_mfovs2 = sorted(indexed_ts2.keys())

    layer1 = indexed_ts1[0][0]["layer"]
    layer2 = indexed_ts2[0][0]["layer"]
    to_ret = []

    # Get all the centers of each section
    log_controller.debug("Fetching sections centers...")
    centers1 = np.array([get_center(indexed_ts1[m]) for m in sorted_mfovs1])
    centers2 = np.array([get_center(indexed_ts2[i]) for i in sorted_mfovs2])

    # Take the mfov closest to the middle of each section
    section_center1 = np.mean(centers1, axis=0)
    section_center2 = np.mean(centers2, axis=0)
    closest_mfovs1_num = 3
    # Find the closest mfovs to the center of section 1
    if num_mfovs1 <= closest_mfovs1_num:
        closest_mfovs_nums1 = indexed_ts1.keys()
    else:
        closest_mfovs_nums1 = np.argpartition([((c[0] - section_center1[0]) ** 2 + (c[1] - section_center1[1]) ** 2)
                                               for c in centers1], closest_mfovs1_num)[:closest_mfovs1_num]
        closest_mfovs_nums1 = [sorted_mfovs1[n] for n in closest_mfovs_nums1]
    # Find the closest mfov to the center of section 2
    centers_mfovs_nums2 = [np.argmin([((c[0] - section_center2[0]) ** 2 +
                                       (c[1] - section_center2[1]) ** 2) for c in centers2])]
    centers_mfovs_nums2 = [sorted_mfovs2[n] for n in centers_mfovs_nums2]

    # Compute per-mfov bounding box for the 2nd section
    section2_mfov_bboxes = {m: BoundingBox.read_bbox_from_ts(indexed_ts2[m].values()) for m in sorted_mfovs2}

    log_controller.debug("Comparing Sec{} (mfovs: {}) and"
                         " Sec{} (starting from mfovs: {})".format(layer1, closest_mfovs_nums1,
                                                                   layer2, centers_mfovs_nums2))
    # Do an iterative search of the mfovs closest to the center of section 1 to the mfovs of section2
    best_transform, num_filtered, \
        filter_rate, _, _, _, \
        initial_search_iters_num = iterative_search(align_args, layer1, layer2, indexed_ts1,
                                                    indexed_ts2, features_folder_path1, features_folder_path2,
                                                    closest_mfovs_nums1, centers_mfovs_nums2, section2_mfov_bboxes,
                                                    sorted_mfovs2, is_initial_search=True)

    if best_transform is None:
        log_controller.debug("Could not find a preliminary transform "
                             "between sections: {} and {}.".format(layer1, layer2))
        return to_ret, initial_search_iters_num

    best_transform_matrix = best_transform.get_matrix()
    log_controller.debug("Found a preliminary transform between sections: {} and {} "
                         "(filtered matches#: {}, rate: {}), with model: {} in and "
                         "{} iterations".format(layer1, layer2, num_filtered, filter_rate,
                                                best_transform_matrix, initial_search_iters_num))

    # Iterate through the mfovs of section1, and try to find
    # for each mfov the transformation to section 2
    # (do an iterative search as was done in the previous phase)
    for i in range(0, num_mfovs1):
        center1 = centers1[i]
        center1_transformed = np.dot(best_transform_matrix, np.append(center1, [1]))[0:2]
        distances = np.array([np.linalg.norm(center1_transformed - centers2[j]) for j in range(num_mfovs2)])
        log_controller.debug("distances:{}".format([str(x) + ":" + str(d) for x, d in enumerate(distances)]))
        relevant_mfovs_nums2 = [sorted_mfovs2[np.argsort(distances)[0]]]
        log_controller.debug("Initial assumption Section {} mfov {} will "
                             "match Section {} mfovs {}".format(layer1, sorted_mfovs1[i], layer2, relevant_mfovs_nums2))
        # Do an iterative search of the mfov from section 1 to the "corresponding" mfov of section2
        mfov_transform, num_filtered, filter_rate, num_rod, num_m1, num_m2, match_iterations = iterative_search(
            align_args, layer1, layer2, indexed_ts1, indexed_ts2,
            features_folder_path1, features_folder_path2, [sorted_mfovs1[i]], relevant_mfovs_nums2,
            section2_mfov_bboxes, sorted_mfovs2, is_initial_search=False)
        if mfov_transform is None:
            # Could not find a transformation for the given mfov
            log_controller.debug("Could not find a transformation between Section {} mfov {}, to Section {} "
                                 ", skipping the mfov".format(layer1, sorted_mfovs1[i], layer2))
        else:
            log_controller.debug("Found a transformation between section {} mfov {} to "
                                 "section {} (filtered matches#: {}, rate: {}), "
                                 "with model: {}".format(layer1, sorted_mfovs1[i], layer2, num_filtered,
                                                         filter_rate, mfov_transform.get_matrix()))
            dict_entry = {
                'mfov1': sorted_mfovs1[i],
                'section2_center': center1_transformed.tolist(),
                'features_in_mfov1': num_m1,
                'transformation':
                    {
                        "className": mfov_transform.class_name,
                        "matrix": mfov_transform.get_matrix().tolist()
                    },
                'matches_rod': num_rod,
                'matches_model': num_filtered,
                'filter_rate': filter_rate,
                'mfov_iterations_num': match_iterations
            }
            to_ret.append(dict_entry)

    return to_ret, initial_search_iters_num


def pre_match_layers(tilespecs_file_path1: str, features_folder_path1: str,
                     tilespecs_file_path2: str, features_folder_path2: str, output_file_path: str, align_args: dict):
    """
    Pre-match two layers using the features
    :param tilespecs_file_path1:
    :param features_folder_path1:
    :param tilespecs_file_path2:
    :param features_folder_path2:
    :param output_file_path:
    :param align_args:
    :return:
    """
    log_controller.debug("Matching layers: {} and {}".format(tilespecs_file_path1, tilespecs_file_path2))
    retval, initial_search_iters_num = analyze_slices(tilespecs_file_path1, tilespecs_file_path2,
                                                      features_folder_path1, features_folder_path2, align_args)

    if len(retval) == 0:
        log_controller.warning("Could not find a match!")
    else:
        jsonfile = {
            'tilespec1': tilespecs_file_path1,
            'tilespec2': tilespecs_file_path2,
            'matches': retval,
            'initial_search_iterations_num': initial_search_iters_num
        }
        with open(output_file_path, 'w') as out:
            json.dump(jsonfile, out, indent=4)
        log_controller.debug("Saved the layers pre-matching result into {}".format(output_file_path))
