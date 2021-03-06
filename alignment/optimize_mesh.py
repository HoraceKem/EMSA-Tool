import sys
import json
import os
import os.path
import numpy as np
import pickle
from scipy.spatial import Delaunay
from sklearn.utils.extmath import randomized_svd
import pylab
from matplotlib import collections as mc
import gc
from common.bounding_box import BoundingBox
from common import utils
import datetime
from alignment import mesh_derivs_multibeam

overall_args = utils.load_json_file('arguments/overall_args.json')
log_controller = utils.LogController('alignment', 'optimize_mesh',
                                     os.path.join(overall_args["base"]["workspace"], 'log'))


sys.setrecursionlimit(10000)  # for grad


class Mesh(object):
    def __init__(self, points):
        # load the mesh
        self.orig_pts = np.array(points, dtype=np.float64).reshape((-1, 2)).copy()
        self.pts = np.array(points, dtype=np.float64).reshape((-1, 2)).copy()
        center = self.pts.mean(axis=0)
        self.pts -= center
        self.pts *= 1.1
        self.pts += center
        self.orig_pts = self.pts.copy()

        log_controller.debug("# points in base mesh {}".format(self.pts.shape[0]))

        # for neighbor searching and internal mesh
        self.triangulation = Delaunay(self.pts)

    def internal_structural_mesh(self):
        simplices = self.triangulation.simplices.astype(np.uint32)
        # find unique edges
        edge_indices = np.vstack((simplices[:, :2],
                                  simplices[:, 1:],
                                  simplices[:, [0, 2]]))
        edge_indices = np.vstack(tuple(sorted(tuple(row)) for row in edge_indices)).astype(np.uint32)
        # mesh.pts[edge_indices, :].shape =(#edges, #pts-per-edge, #values-per-pt)
        edge_lengths = np.sqrt((np.diff(self.pts[edge_indices], axis=1) ** 2).sum(axis=2)).ravel()
        triangles_as_pts = self.pts[simplices]
        triangle_areas = 0.5 * np.cross(triangles_as_pts[:, 2, :] - triangles_as_pts[:, 0, :],
                                        triangles_as_pts[:, 1, :] - triangles_as_pts[:, 0, :])
        return edge_indices, edge_lengths, simplices, triangle_areas

    def remove_unneeded_points(self, pts1, pts2):
        """Removes points that cause query_cross_barycentrics to fail"""
        p1 = pts1.copy()
        p1[p1 < 0] = 0.01
        simplex_indices = self.triangulation.find_simplex(p1)
        if np.any(simplex_indices == -1):
            locs = np.where(simplex_indices == -1)
            log_controller.debug("locations:{}".format(locs))
            log_controller.debug("points:{}".format(pts1[locs]))
            log_controller.debug("removing the above points")
            pts1 = np.delete(pts1, locs, 0)
            pts2 = np.delete(pts2, locs, 0)
        return pts1, pts2

    def query_barycentrics(self, points):
        """Returns the mesh indices that surround a point, and the barycentric weights of those points"""
        p = points.copy()
        p[p < 0] = 0.01
        simplex_indices = self.triangulation.find_simplex(p)
        assert not np.any(simplex_indices == -1)

        # http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        x = self.triangulation.transform[simplex_indices, :2]
        y = points - self.triangulation.transform[simplex_indices, 2]
        b = np.einsum('ijk,ik->ij', x, y)
        pt_indices = self.triangulation.simplices[simplex_indices].astype(np.uint32)
        barys = np.c_[b, 1 - b.sum(axis=1)]

        return self.triangulation.simplices[simplex_indices].astype(np.uint32), barys


def linearize_grad(positions, gradients):
    """
    Perform a least-squares fit, then return the values from that fit
    :param positions:
    :param gradients:
    :return:
    """
    positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
    XTX = np.dot(positions.T, positions)
    XTY = np.dot(positions.T, gradients)
    beta = np.dot(np.linalg.inv(XTX), XTY)
    return np.dot(positions, beta)


def blend(a, b, t):
    """
    at t=0, return a, at t=1, return b
    :param a:
    :param b:
    :param t:
    :return:
    """
    return a + (b - a) * t


def mean_offsets(meshes, links, stop_at_ts, plot=False):
    means = []
    for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.items():
        if ts1 > stop_at_ts or ts2 > stop_at_ts:
            continue
        pts1 = np.einsum('ijk,ij->ik', meshes[ts1].pts[idx1], w1)
        pts2 = np.einsum('ijk,ij->ik', meshes[ts2].pts[idx2], w2)
        if plot:
            lines = [[p1, p2] for p1, p2 in zip(pts1, pts2)]

            lc = mc.LineCollection(lines)
            pylab.figure()
            pylab.title(ts1 + ' ' + ts2)
            pylab.gca().add_collection(lc)
            pylab.scatter(pts1[:, 0], pts1[:, 1])
            pylab.gca().autoscale()

        lens = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
        means.append(np.median(lens))
    return np.mean(means)


def ts_mean_offsets(meshes, links, ts, plot=False):
    means = []
    for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.items():
        if ts not in (ts1, ts2):
            continue
        pts1 = np.einsum('ijk,ij->ik', meshes[ts1].pts[idx1], w1)
        pts2 = np.einsum('ijk,ij->ik', meshes[ts2].pts[idx2], w2)
        if plot:
            lines = [[p1, p2] for p1, p2 in zip(pts1, pts2)]

            lc = mc.LineCollection(lines)
            pylab.figure()
            pylab.title(ts1 + ' ' + ts2)
            pylab.gca().add_collection(lc)
            pylab.scatter(pts1[:, 0], pts1[:, 1])
            pylab.gca().autoscale()

        lens = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
        means.append(np.median(lens))
    return np.mean(means)


def plot_offsets(meshes, links, ts, fname_prefix):
    for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.items():
        if ts1 != ts:
            continue
        pts1 = np.einsum('ijk,ij->ik', meshes[ts1].pts[idx1], w1)
        pts2 = np.einsum('ijk,ij->ik', meshes[ts2].pts[idx2], w2)
        diffs2 = pts2 - pts1

        figure_file_path = "{}{}.png".format(fname_prefix, (os.path.basename(ts1) + '_' +
                                                            os.path.basename(ts2)).replace(' ', '_'))
        pylab.figure()
        pylab.title(ts1 + ' ' + ts2)
        pylab.quiver(pts1[:, 0], pts1[:, 1], diffs2[:, 0], diffs2[:, 1])
        pylab.savefig(figure_file_path)


def plot_points(pts, fname):
    pylab.figure()
    pylab.scatter(pts[:, 0], pts[:, 1])
    pylab.savefig(fname)


def align_rigid(pts1, pts2):
    # find R,T that bring pts1 to pts2

    # convert to column vectors
    pts1 = pts1.T
    pts2 = pts2.T

    m1 = pts1.mean(axis=1, keepdims=True)
    m2 = pts2.mean(axis=1, keepdims=True)
    U, S, VT = np.linalg.svd(np.dot((pts1 - m1), (pts2 - m2).T))
    R = np.dot(VT.T, U.T)
    T = - np.dot(R, m1) + m2

    # convert to form used for left-multiplying row vectors
    return R.T, T.T


def Haffine_from_points(fp, tp):
    """
    Find H, affine transformation, s.t. tp is affine transformation of fp.
    Taken from 'Programming Computer Vision with Python: Tools and algorithms for analyzing images'
    """
    assert(fp.shape == tp.shape)
    fp = fp.T
    tp = tp.T

    # condition points
    # --- from points ---
    m = fp.mean(axis=1, keepdims=True)
    maxstd = np.max(np.std(fp, axis=1)) + 1e-9
    C1 = np.diag([1.0/maxstd, 1.0/maxstd, 1.0])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = np.dot(C1[:2, :2], fp) + np.asarray(C1[:2, 2]).reshape((2, 1))

    # --- to points ---
    m = tp.mean(axis=1, keepdims=True)
    C2 = C1.copy()  # must use same scaling for both point sets
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = np.dot(C2[:2, :2], tp) + np.asarray(C2[:2, 2]).reshape((2, 1))

    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond, tp_cond), axis=0)
    # U,S,V = np.linalg.svd(A.T, full_matrices=True)
    # U,S,V = spLA.svd(A.T)
    # U,S,V = spLA.svds(A.T, k=3, ncv=50, maxiter=1e9, return_singular_vectors='vh') # doesn't work well
    # Randomized svd uses much less memory and is much faster than the numpy/scipy versions
    U, S, V = randomized_svd(A.T, 4, random_state=None)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    m = H / H[2][2]
    return m[:2, :2].T, m[:2, 2]  # rotation part (transposed) and translation


def get_transform_matrix(pts1, pts2, model_type):
    if model_type == 1:
        return align_rigid(pts1, pts2)
    elif model_type == 3:
        return Haffine_from_points(pts1, pts2)
    else:
        log_controller.debug("Unsupported transformation model type")
        return None


def optimize_meshes_links(meshes, links, layers, align_args):
    log_controller.debug("Debugged layers: {}".format(align_args["optimize_mesh"]["debugged_layers"]))
    DEBUG_DIR = None
    if len(align_args["optimize_mesh"]["debugged_layers"]) > 0:
        align_args["optimize_mesh"]["debugged_layers"] = map(int,
                                                             align_args["optimize_mesh"]["debugged_layers"].split(','))
        DEBUG_DIR = os.path.join("debug_optimization", "logs_{}".format(datetime.datetime.now().isoformat()))
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)

    # Build internal structural mesh
    # (edge_indices, edge_lengths, face_indices, face_areas)
    structural_meshes = {ts: mesh.internal_structural_mesh() for ts, mesh in meshes.items()}

    # pre-optimization: rigid alignment of slices
    # TODO - sort by the layer and not by the meshes keys
    sorted_slices = sorted(meshes.keys())

    # TODO - make this faster by just iterating over the debugged layers
    for active_ts in sorted_slices:
        if layers[active_ts] in align_args["optimize_mesh"]["debugged_layers"]:
            postfix = os.path.basename(active_ts).replace(' ', '_')
            pickle.dump([active_ts, meshes[active_ts].pts],
                        open(os.path.join(DEBUG_DIR, "pre_affine_{}.pickle".format(postfix)), "w"))

    for active_ts in sorted_slices[1:]:
        log_controller.debug("Before affine (sec {}): {}".format(layers[active_ts],
                                                                 mean_offsets(meshes, links, active_ts, plot=False)))
        rot = 0
        tran = 0
        count = 0
        for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.items():
            if active_ts in (ts1, ts2) and (ts1 <= active_ts) and (ts2 <= active_ts):
                pts1 = np.einsum('ijk,ij->ik', meshes[ts1].pts[idx1], w1)
                pts2 = np.einsum('ijk,ij->ik', meshes[ts2].pts[idx2], w2)
                if ts1 == active_ts:
                    cur_rot, cur_tran = get_transform_matrix(pts1, pts2, align_args["optimize_mesh"]["assumed_model"])
                else:
                    cur_rot, cur_tran = get_transform_matrix(pts2, pts1, align_args["optimize_mesh"]["assumed_model"])
                # Average the affine transformation by the number of matches between the two sections
                rot += pts1.shape[0] * cur_rot
                tran += pts1.shape[0] * cur_tran
                count += pts1.shape[0]
        if count == 0:
            log_controller.debug("Error: no matches found for section {}.".format(active_ts))
            sys.exit(1)
        # meshes[active_ts].pts = np.dot(meshes[active_ts].pts, rot / count) + (trans / count)
        # normalize the transformation
        rot = rot * (1.0 / count)
        tran = tran * (1.0 / count)
        # transform the points
        meshes[active_ts].pts = np.dot(meshes[active_ts].pts, rot) + tran
        log_controller.debug("After affine (sec {}): {}\n".format(layers[active_ts],
                                                                  mean_offsets(meshes, links, active_ts, plot=False)))

    # TODO - make this faster by just iterating over the debugged layers
    for active_ts in sorted_slices:
        if layers[active_ts] in align_args["optimize_mesh"]["debugged_layers"]:
            postfix = os.path.basename(active_ts).replace(' ', '_')
            pickle.dump([active_ts, meshes[active_ts].pts],
                        open(os.path.join(DEBUG_DIR, "post_affine_{}.pickle".format(postfix)), "w"))

    log_controller.debug("After pre-opt MO: {}\n".format(mean_offsets(meshes, links, sorted_slices[-1], plot=False)))

    stepsize = 0.0001
    momentum = 0.5
    prev_cost = np.inf
    gradients_with_momentum = {ts: 0.0 for ts in meshes}
    old_pts = None

    for i in range(align_args["optimize_mesh"]["max_iterations"]):
        cost = 0.0

        gradients = {ts: np.zeros_like(mesh.pts) for ts, mesh in meshes.items()}

        # Compute the cost of the internal and external links
        for ts in meshes:
            cost += mesh_derivs_multibeam.internal_grad(meshes[ts].pts, gradients[ts],
                                                        *((structural_meshes[ts]) +
                                                          (align_args["optimize_mesh"]["intra_slice_weight"],
                                                           align_args["optimize_mesh"]["intra_slice_winsor"])))
        for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.items():
            cost += mesh_derivs_multibeam.external_grad(meshes[ts1].pts, meshes[ts2].pts,
                                                        gradients[ts1], gradients[ts2],
                                                        idx1, w1,
                                                        idx2, w2,
                                                        align_args["optimize_mesh"]["cross_slice_weight"] /
                                                        float(abs(layers[ts1] - layers[ts2])),
                                                        align_args["optimize_mesh"]["cross_slice_winsor"])

        if cost < prev_cost and not np.isinf(cost):
            prev_cost = cost
            stepsize *= 1.1
            if stepsize > 1.0:
                stepsize = 1.0
            # update with new gradients
            for ts in gradients_with_momentum:
                gradients_with_momentum[ts] = gradients[ts] + momentum * gradients_with_momentum[ts]
            old_pts = {ts: m.pts.copy() for ts, m in meshes.items()}
            for ts in meshes:
                meshes[ts].pts -= stepsize * gradients_with_momentum[ts]
        else:  # we took a bad step: undo it, scale down stepsize, and start over
            for ts in meshes:
                meshes[ts].pts = old_pts[ts]
            stepsize *= 0.5
            gradients_with_momentum = {ts: 0 for ts in meshes}
        if i % 100 == 0:
            log_controller.debug("i {}: C: {}, MO: {}, S: {}".format(i, cost,
                                                                     mean_offsets(meshes, links, sorted_slices[-1],
                                                                                  plot=False), stepsize))

        # Save animation for debugging
        if len(align_args["optimize_mesh"]["debugged_layers"]) > 0:
            # TODO - make this faster by just iterating over the debugged layers
            for active_ts in sorted_slices:
                if layers[active_ts] in align_args["optimize_mesh"]["debugged_layers"]:
                    postfix = os.path.basename(active_ts).replace(' ', '_')
                    pickle.dump([active_ts, meshes[active_ts].pts],
                                open(os.path.join(DEBUG_DIR, "post_iter{}_{}.pickle".format(str(i).zfill(5),
                                                                                            postfix)), "w"))

        for ts in meshes:
            assert not np.any(~ np.isfinite(meshes[ts].pts))

        # If stepsize is too small (won't make any difference), stop the iterations
        if stepsize < align_args["optimize_mesh"]["min_stepsize"]:
            break
    log_controller.debug("last MO: {}\n".format(mean_offsets(meshes, links, sorted_slices[-1], plot=False)))

    log_controller.debug("Final MO:")
    for active_ts in sorted_slices:
        log_controller.debug(" Section {}: {}".format(layers[active_ts], ts_mean_offsets(meshes, links,
                                                                                         active_ts, plot=False)))

    # Prepare per-layer output
    out_positions = {}

    for i, ts in enumerate(meshes):
        out_positions[ts] = [meshes[ts].orig_pts,
                             meshes[ts].pts]
        if "DUMP_LOCATIONS" in os.environ:
            pickle.dump([ts, out_positions[ts]], open("newpos{}.pickle".format(i), "w"))

    return out_positions


def optimize_meshes(match_files_list, hex_spacing, align_args):
    meshes = {}

    meshes_per_mfov = {}
    pair_ts_to_pts = {}
    layers = {}
    # extract meshes
    for match_file in match_files_list:
        # Assumes that the mesh can be separated into multiple files with the same tilespec1 or tilespec2
        data = None
        log_controller.debug(match_file)
        with open(match_file, 'r') as f:
            data = json.load(f)
        if not data["tilespec1"] in meshes:
            tilespecs_file_path = data["tilespec1"].replace("file://", "")
            log_controller.debug("Generating Hexagonal Grid")
            bb = BoundingBox.read_bbox(tilespecs_file_path)
            log_controller.debug("bounding_box: {}".format(bb))
            meshes[data["tilespec1"]] = Mesh(utils.generate_hexagonal_grid(bb, hex_spacing))
            ts_layer = utils.read_layer_from_tilespecs_file(tilespecs_file_path)
            layers[data["tilespec1"]] = ts_layer
        if not data["tilespec1"] in pair_ts_to_pts:
            pair_ts_to_pts[data["tilespec1"]] = {}
        if not data["tilespec2"] in pair_ts_to_pts[data["tilespec1"]]:
            pair_ts_to_pts[data["tilespec1"]][data["tilespec2"]] = []
        pts1 = np.array([p["point1"] for p in data["pointmatches"]])
        pts2 = np.array([p["point2"] for p in data["pointmatches"]])
        if len(pts1) > 0:
            pair_ts_to_pts[data["tilespec1"]][data["tilespec2"]].append((pts1, pts2))

    links = {}
    # extract links
    for ts1 in meshes:
        for ts2 in pair_ts_to_pts[ts1]:
            pts1 = np.concatenate([l[0] for l in pair_ts_to_pts[ts1][ts2]])
            pts2 = np.concatenate([l[1] for l in pair_ts_to_pts[ts1][ts2]])

            if len(pts1) > 0:
                pts1, pts2 = meshes[ts1].remove_unneeded_points(pts1, pts2)
                pts2, pts1 = meshes[ts2].remove_unneeded_points(pts2, pts1)

                links[ts1, ts2] = (meshes[ts1].query_barycentrics(pts1),
                                   meshes[ts2].query_barycentrics(pts2))

    # free some memory
    del pair_ts_to_pts
    gc.collect()

    return optimize_meshes_links(meshes, links, layers, align_args)
