import numpy as np
import h5py
import ransac
import json
from common.trans_models import Transforms
from common.bounding_box import BoundingBox
import common.keypoint_features_matching as matching


def load_features(h5_file_path):
    with h5py.File(h5_file_path, 'r') as hf:
        image_url = str(hf["imageUrl"][...]).replace('b\'', '').replace('\'', '')
        locations = hf["pts/locations"][...]
        descriptors = hf["descs"][...]
    return image_url, locations, descriptors


def get_tile_specification_transformation(tile_specification):
    transforms = tile_specification["transforms"]
    transform = Transforms.from_tilespec(transforms[0])
    return transform


def match_and_save(features1_path, features2_path, _ts1, _ts2, save_match_path, parameters, arguments):
    image1_path, pts1, des1 = load_features(features1_path)
    image2_path, pts2, des2 = load_features(features2_path)
    cur_bbox1 = BoundingBox.fromList(_ts1["bbox"])
    cur_bbox2 = BoundingBox.fromList(_ts2["bbox"])
    overlap_bbox = cur_bbox1.intersect(cur_bbox2).expand(offset=50)
    tilespec1_transform = get_tile_specification_transformation(_ts1)
    tilespec2_transform = get_tile_specification_transformation(_ts2)
    features_mask1 = overlap_bbox.contains(tilespec1_transform.apply(pts1))
    features_mask2 = overlap_bbox.contains(tilespec2_transform.apply(pts2))

    pts1 = pts1[features_mask1]
    pts2 = pts2[features_mask2]
    des1 = des1[features_mask1]
    des2 = des2[features_mask2]
    good_matches = matching.__dict__[arguments.match](des1, des2, parameters)
    if good_matches == ['NO_MATCH']:
        filtered_matches = [[], []]
        model_json = []
    else:
        match_points = np.array([
            np.array([pts1[[m[0].queryIdx for m in good_matches]]][0]),
            np.array([pts2[[m[0].trainIdx for m in good_matches]]][0])
        ])

        model, filtered_matches, _, _ = ransac.filter_matches(match_points,
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
        else:
            model_json = model.to_modelspec()

    out_data = [{
        "mipmapLevel": 0,
        "url1": image1_path,
        "url2": image2_path,
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

    print("Saving matches into {}".format(save_match_path))
    with open(save_match_path, 'w') as out:
        json.dump(out_data, out, sort_keys=True, indent=4)
