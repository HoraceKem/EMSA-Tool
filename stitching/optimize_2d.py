import json
import os
import random
import numpy as np
import scipy.sparse as spp
from common import utils
from scipy.sparse.linalg import lsqr
from collections import defaultdict

overall_args = utils.load_json_file('arguments/overall_args.json')
utils.create_dir(overall_args["base"]["workspace"])
log_controller = utils.LogController('stitching', 'optimize_2d', os.path.join(overall_args["base"]["workspace"], 'log'))


def find_rotation(point1: np.ndarray, point2: np.ndarray, step_size: float):
    """
    Find rotation
    :param point1:
    :param point2:
    :param step_size:
    :return:
    """
    U, S, VT = np.linalg.svd(np.dot(point1, point2.T))
    R = np.dot(VT.T, U.T)
    angle = step_size * np.arctan2(R[1, 0], R[0, 0])
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def create_new_tilespec(old_tilespecs_file_path: str, rotations: dict, translations: dict,
                        centers: dict, new_tilespecs_file_path: str):
    """
    Load original tilespecs and write the transforms into it
    :param old_tilespecs_file_path:
    :param rotations:
    :param translations:
    :param centers:
    :param new_tilespecs_file_path:
    :return:
    """
    log_controller.debug("Optimization done, saving tilespec at: {}".format(new_tilespecs_file_path))
    tilespecs = utils.load_json_file(old_tilespecs_file_path)

    for ts in tilespecs:
        img_url = ts["mipmapLevels"]["0"]["imageUrl"]
        if img_url not in rotations.keys():
            log_controller.debug("Flagging out tile {}, as no rotation was found".format(img_url))
            continue

        old_bbox = [float(d) for d in ts["bbox"]]
        old_bbox_points = [
            np.array([np.array([old_bbox[0]]), np.array([old_bbox[2]])]),
            np.array([np.array([old_bbox[1]]), np.array([old_bbox[2]])]),
            np.array([np.array([old_bbox[0]]), np.array([old_bbox[3]])]),
            np.array([np.array([old_bbox[1]]), np.array([old_bbox[3]])])
        ]

        trans = np.array(translations[img_url])  # an array of 2 elements
        rot_matrix = np.matrix(rotations[img_url]).T  # a 2x2 matrix
        center = np.array(centers[img_url])  # an array of 2 elements
        transformed_points = [np.dot(rot_matrix, old_point - center) + center + trans for old_point in old_bbox_points]

        min_xy = np.min(transformed_points, axis=0).flatten()
        max_xy = np.max(transformed_points, axis=0).flatten()
        new_bbox = [min_xy[0], max_xy[0], min_xy[1], max_xy[1]]

        delta = np.asarray(transformed_points[0].T)[0]
        x, y = np.asarray((old_bbox_points[1] - old_bbox_points[0]).T)[0]
        new_x, new_y = np.asarray(transformed_points[1].T)[0]
        k = (y * (new_x - delta[0]) - x * (new_y - delta[1])) / (x ** 2 + y ** 2)
        h1 = (new_x - delta[0] - k * y) / x

        if h1 > 1.0:
            h1 = 1.0
        new_transformation = "{} {} {}".format(np.arccos(h1), delta[0], delta[1])

        ts["transforms"] = [{
            "className": "mpicbg.trakem2.transform.RigidModel2D",
            "dataString": new_transformation
        }]

        ts["bbox"] = new_bbox

    with open(new_tilespecs_file_path, 'w') as f:
        json.dump(tilespecs, f, sort_keys=True, indent=4)
        log_controller.debug('Wrote tilespec to {}'.format(new_tilespecs_file_path))


def optimize_2d_stitching(tilespecs_file_path: str, match_file_paths: list, output_json_file_path: str):
    """
    Optimize the global stitching according to matching pairs
    :param tilespecs_file_path:
    :param match_file_paths:
    :param output_json_file_path:
    :return:
    """
    all_matches = {}
    all_points = defaultdict(list)

    max_iters = 1000
    epsilon = 5
    step_size = 0.1
    damping = 0.01
    no_empty_matches_flag = True
    tile_specification = utils.load_json_file(tilespecs_file_path)

    for f in match_file_paths:
        data = json.load(open(f))
        points1 = np.array([c["p1"]["w"] for c in data[0]["correspondencePointPairs"]]).T
        points2 = np.array([c["p2"]["w"] for c in data[0]["correspondencePointPairs"]]).T
        url1 = data[0]["url1"]
        url2 = data[0]["url2"]
        log_controller.debug("url1: {}, url2: {}".format(url1, url2))

        if points1.size > 0:
            all_matches[url1, url2] = (points1, points2)
            all_points[url1].append(points1)
            all_points[url2].append(points2)
        elif no_empty_matches_flag:
            tile1 = {}
            tile2 = {}
            for t in tile_specification:
                if t["mipmapLevels"]["0"]["imageUrl"] == url1:
                    tile1 = t
                if t["mipmapLevels"]["0"]["imageUrl"] == url2:
                    tile2 = t

            overlap_x_min = max(tile1["bbox"][0], tile2["bbox"][0])
            overlap_x_max = min(tile1["bbox"][1], tile2["bbox"][1])
            overlap_y_min = max(tile1["bbox"][2], tile2["bbox"][2])
            overlap_y_max = min(tile1["bbox"][3], tile2["bbox"][3])
            overlap_bbox = [overlap_x_min, overlap_x_max, overlap_y_min, overlap_y_max]
            x_range, y_range = overlap_bbox[1] - overlap_bbox[0], overlap_bbox[3] - overlap_bbox[2]
            if x_range < 0 or y_range < 0:
                continue

            x_values, y_values = [], []
            x_values.append(random.random() * x_range / 2 + overlap_bbox[0])
            x_values.append(random.random() * x_range / 2 + overlap_bbox[0] + x_range / 2)
            x_values.append(random.random() * x_range / 2 + overlap_bbox[0])
            x_values.append(random.random() * x_range / 2 + overlap_bbox[0] + x_range / 2)

            y_values.append(random.random() * y_range / 2 + overlap_bbox[2])
            y_values.append(random.random() * y_range / 2 + overlap_bbox[2])
            y_values.append(random.random() * y_range / 2 + overlap_bbox[2] + y_range / 2)
            y_values.append(random.random() * y_range / 2 + overlap_bbox[2] + y_range / 2)

            correspondence_pairs = []
            for i in range(0, len(x_values)):
                new_pair = {"dist_after_ransac": 1.0,
                            "p1": {"l": [x_values[i] - tile1["bbox"][0], y_values[i] - tile1["bbox"][2]],
                                   "w": [x_values[i], y_values[i]]},
                            "p2": {"l": [x_values[i] - tile2["bbox"][0], y_values[i] - tile2["bbox"][2]],
                                   "w": [x_values[i], y_values[i]]}}
                correspondence_pairs.append(new_pair)

            points1 = np.array([c["p1"]["w"] for c in correspondence_pairs]).T
            points2 = np.array([c["p2"]["w"] for c in correspondence_pairs]).T
            all_matches[url1, url2] = (points1, points2)
            all_points[url1].append(points1)
            all_points[url2].append(points2)

    centers = {k: np.mean(np.hstack(points), axis=1, keepdims=True) for k, points in all_points.items()}
    url_index = {url: index for index, url in enumerate(all_points)}

    prev_mean_med = np.inf

    T = defaultdict(lambda: np.zeros((2, 1)))
    R = defaultdict(lambda: np.eye(2))

    for i in range(max_iters):
        # transform points by the current trans/rot
        trans_matches = {(k1, k2): (np.dot(R[k1], p1 - centers[k1]) + T[k1] + centers[k1],
                                    np.dot(R[k2], p2 - centers[k2]) + T[k2] + centers[k2])
                         for (k1, k2), (p1, p2) in all_matches.items()}

        # mask off all points more than epsilon past the median
        diffs = {k: p2 - p1 for k, (p1, p2) in trans_matches.items()}
        distances = {k: np.sqrt((d ** 2).sum(axis=0)) for k, d in diffs.items()}
        masks = {k: d < (np.median(d) + epsilon) for k, d in distances.items()}
        masked_matches = {k: (p1[:, masks[k]], p2[:, masks[k]]) for k, (p1, p2) in trans_matches.items()}

        median_dists = [np.median(d) for d in distances.values()]
        med_med = np.median(median_dists)
        mean_med = np.mean(median_dists)
        max_med = np.max(median_dists)
        log_controller.debug("med-med distance: {}, mean-med distance: {}  "
                             "max-med: {}  SZ: {}".format(med_med, mean_med, max_med, step_size))
        if mean_med < prev_mean_med:
            step_size *= 1.1
            if step_size > 1:
                step_size = 1
        else:
            step_size *= 0.5
        prev_mean_med = mean_med

        rows = np.hstack((np.arange(len(diffs)), np.arange(len(diffs))))
        cols = np.hstack(([url_index[url1] for (url1, url2) in diffs],
                          [url_index[url2] for (url1, url2) in diffs]))

        # diffs are p2 - p1, so we want a positive value on the translation for p1,
        # e.g., a solution could be Tp1 == p2 - p1.

        M_vals = np.hstack(([pts.shape[1] for pts in diffs.values()],
                            [-pts.shape[1] for pts in diffs.values()]))

        M = spp.csr_matrix((M_vals, (rows, cols)))

        # We use the sum of match differences
        D = np.vstack([d.sum(axis=1) for d in diffs.values()])
        oTx = lsqr(M, D[:, :1], damp=damping)[0]
        oTy = lsqr(M, D[:, 1:], damp=damping)[0]
        for k, idx in url_index.items():
            T[k][0] += oTx[idx]
            T[k][1] += oTy[idx]

        # first iteration is translation only
        if i == 0:
            continue

        # don't update Rotations on last iteration
        if step_size < 1e-30:
            log_controller.debug("Step size is small enough, finishing optimization")
            break

        # don't update Rotations on last iteration
        if i < max_iters - 1:
            self_points = defaultdict(list)
            other_points = defaultdict(list)
            for (k1, k2), (p1, p2) in masked_matches.items():
                self_points[k1].append(p1)
                self_points[k2].append(p2)
                other_points[k1].append(p2)
                other_points[k2].append(p1)
            self_points = {k: np.hstack(p) for k, p in self_points.items()}
            other_points = {k: np.hstack(p) for k, p in other_points.items()}

            self_centers = {k: np.mean(p, axis=1).reshape((2, 1)) for k, p in self_points.items()}
            other_centers = {k: np.mean(p, axis=1).reshape((2, 1)) for k, p in other_points.items()}

            new_R = {k: find_rotation(self_points[k] - self_centers[k],
                                      other_points[k] - other_centers[k],
                                      step_size)
                     for k in self_centers}
            R = {k: np.dot(R[k], new_R[k]) for k in R}

    R = {k: v.tolist() for k, v in R.items()}
    T = {k: v.tolist() for k, v in T.items()}
    centers = {k: v.tolist() for k, v in centers.items()}

    create_new_tilespec(tilespecs_file_path, R, T, centers, output_json_file_path)
