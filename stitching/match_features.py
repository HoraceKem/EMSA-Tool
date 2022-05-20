import numpy as np
import h5py
import json
import os
import common.utils as utils
import common.ransac as ransac
from common.trans_models import Transforms
from common.bounding_box import BoundingBox
import common.keypoint_features_matching as matching


overall_args = utils.load_json_file('arguments/overall_args.json')
log_controller = utils.LogController('stitching', 'match_features',
                                     os.path.join(overall_args["base"]["workspace"], 'log'))


def load_features(h5_file_path: str) -> [np.array, np.array]:
    """
    Load features from h5py file
    :param h5_file_path:
    :type h5_file_path: str
    :return:
    """
    log_controller.debug('Loading features from file: {}'.format(h5_file_path))
    with h5py.File(h5_file_path, 'r') as f:
        image_url = str(f["imageUrl"][...]).replace('b\'', '').replace('\'', '')
        log_controller.debug('The features are extracted from: {}'.format(image_url))
        locations = f["pts/locations"][...]
        descriptors = f["descs"][...]
        f.close()
    return locations, descriptors


def get_tile_specification_transformation(tilespec: dict):
    """
    Get transform in 'trans_models' from tile specification
    :param tilespec:
    :return:
    """
    transforms = tilespec["transforms"]
    transform = Transforms.from_tilespec(transforms[0])
    return transform


def match_and_save(features1_path: str, features2_path: str, ts1: dict, ts2: dict,
                   save_match_path: str, parameters: dict, matching_type: str):
    """
    Given two features file(h5py format), match them and save the matched into json file
    :param features1_path:
    :param features2_path:
    :param ts1:
    :param ts2:
    :param save_match_path:
    :param parameters:
    :param matching_type:
    :return:
    """
    log_controller.debug('Matching the features from {} and {} using {}'.format(features1_path, features2_path,
                                                                                matching_type))
    if os.path.exists(save_match_path):
        log_controller.debug('Matching file {} exists, skipped.'.format(save_match_path))
        return
    if matching_type not in matching.__dict__:
        log_controller.error('Unexpected matching type. '
                                          'Please refer to common/keypoint_features_matching.py')
        raise TypeError('matching type')
    pts1, des1 = load_features(features1_path)
    pts2, des2 = load_features(features2_path)
    cur_bbox1 = BoundingBox.fromList(ts1["bbox"])
    cur_bbox2 = BoundingBox.fromList(ts2["bbox"])
    overlap_bbox = cur_bbox1.intersect(cur_bbox2).expand(offset=50)
    log_controller.debug('Overlapping bounding box: {}'.format(overlap_bbox))
    tilespec1_transform = get_tile_specification_transformation(ts1)
    tilespec2_transform = get_tile_specification_transformation(ts2)
    features_mask1 = overlap_bbox.contains(tilespec1_transform.apply(pts1))
    features_mask2 = overlap_bbox.contains(tilespec2_transform.apply(pts2))

    pts1 = pts1[features_mask1]
    pts2 = pts2[features_mask2]
    log_controller.debug('Image 1 features number: {}'.format(len(pts1)))
    log_controller.debug('Image 2 features number: {}'.format(len(pts2)))
    des1 = des1[features_mask1]
    des2 = des2[features_mask2]
    good_matches = matching.__dict__[matching_type](des1, des2, parameters)
    if good_matches == ['NO_MATCH']:
        log_controller.debug('No match between {} and {}'.format(features1_path, features2_path))
        filtered_matches = [[], []]
        model_json = []
    else:
        match_points = np.array([
            np.array([pts1[[m[0].queryIdx for m in good_matches]]][0]),
            np.array([pts2[[m[0].trainIdx for m in good_matches]]][0])
        ])
        log_controller.debug('Detected {} matches'.format(len(good_matches)))
        model, filtered_matches = ransac.filter_matches(match_points,
                                                        parameters["ransac"]["model_index"],
                                                        parameters["ransac"]["iterations"],
                                                        parameters["ransac"]["max_epsilon"],
                                                        parameters["ransac"]["min_inlier_ratio"],
                                                        parameters["ransac"]["min_num_inlier"],
                                                        parameters["ransac"]["max_trust"],
                                                        parameters["ransac"]["del_delta"])
        model_json = []
        if model is None:
            filtered_matches = [[], []]
            log_controller.debug('No fit model between {} and {}'.format(features1_path, features2_path))
        else:
            model_json = model.to_modelspec()
            log_controller.debug('Fit model: {}'.format(model_json))

    out_data = [{
        "mipmapLevel": 0,
        "url1": ts1["mipmapLevels"]["0"]["imageUrl"],
        "url2": ts2["mipmapLevels"]["0"]["imageUrl"],
        "correspondencePointPairs": [
            {
                "p1": {"w": np.array(tilespec1_transform.apply(p1)[:2]).tolist(),
                       "l": np.array([p1[0], p1[1]]).tolist()},
                "p2": {"w": np.array(tilespec2_transform.apply(p2)[:2]).tolist(),
                       "l": np.array([p2[0], p2[1]]).tolist()},
            } for p1, p2 in zip(filtered_matches[0], filtered_matches[1])
        ],
        "model": model_json
    }]

    log_controller.debug("Saving matches into {}".format(save_match_path))
    with open(save_match_path, 'w') as out:
        json.dump(out_data, out, sort_keys=True, indent=4)
