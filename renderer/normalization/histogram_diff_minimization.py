import sys
import os
import cv2
import numpy as np
import ujson as json
from rh_renderer import models
from collections import defaultdict
import scipy.optimize
from lru import LRU
import itertools
import cPickle as pickle
from scipy import spatial


class HistogramDiffMinimization(object):
    __CELL_WIDTH = 64
    __CELL_HEIGHT = 64
    __HALF_CELL_WIDTH = int(__CELL_WIDTH // 2)
    __HALF_CELL_HEIGHT = int(__CELL_HEIGHT // 2)

    def __init__(self, ts_fname):
        self._optimized_ts_fname = ts_fname
        self._scale_map = None

    def adjust_histogram(self, img_path, img):
        hist_scale = self._scale_map[img_path]
        img_scaled = img * hist_scale
        img_scaled[img_scaled > 255] = 255
        
        return img_scaled.astype(img.dtype)


    def _parse_tilespec_fname(self):
        with open(self._optimized_ts_fname, 'r') as f:
            tilespec = json.load(f)

        assert(len(tilespec) > 0)

        img_shape = (tilespec[0]["height"], tilespec[0]["width"]) # height, width

        fnames_transforms_bbox = [[tile_ts["mipmapLevels"]["0"]["imageUrl"].replace("file://", ""), models.Transforms.from_tilespec(tile_ts["transforms"][0]), tile_ts["bbox"]]
                    for tile_ts in tilespec]

        return img_shape, fnames_transforms_bbox

    @staticmethod
    def _get_general_image_grid_cells_centers(img_shape):
        """
        Returns the centers locations of cells in a grid on a general image of the given shape (which should be the same for all images)
        """
        start_x = HistogramDiffMinimization.__HALF_CELL_WIDTH
        start_y = HistogramDiffMinimization.__HALF_CELL_HEIGHT
        image_grid_centers_XY = np.mgrid[start_x:img_shape[1]:HistogramDiffMinimization.__CELL_WIDTH, start_y:img_shape[0]:HistogramDiffMinimization.__CELL_HEIGHT]
        return np.vstack((np.ravel(image_grid_centers_XY[0]), np.ravel(image_grid_centers_XY[1]))).T
        

    @staticmethod
    def _divide_section_images_to_grid_cells(fnames_transforms, general_image_grid_centers):
        """
        Populates a map between a global section grid centers locations and the corresponding images and their local (general) cell centers.

        Returns the grid that maps between the global centers locations to a list of tuples (img_idx, img_local_center),
        and a list of keys to grid cells that have more than one image mapped.
        """
        #section_grid = defaultdict(lambda  : defaultdict(list))
        section_grid = defaultdict(list)
        overlapping_grid_cells_list = []

        for img_idx, (img_fname, transform) in enumerate(fnames_transforms):
            # Apply the transformation to the image grid cells centers
            img_cells_centers = transform.apply(general_image_grid_centers)
            for img_cell_center, orig_center in zip(img_cells_centers, general_image_grid_centers):
                global_img_cell_grid_entry = (int(img_cell_center[0] // HistogramDiffMinimization.__CELL_WIDTH), int(img_cell_center[1] // HistogramDiffMinimization.__CELL_HEIGHT))
                section_grid[global_img_cell_grid_entry].append((img_idx, tuple(orig_center)))
                # Append that entry to the list if it has an overlap between two images
                if len(section_grid[global_img_cell_grid_entry]) == 2:
                    overlapping_grid_cells_list.append(global_img_cell_grid_entry)

        return section_grid, overlapping_grid_cells_list

    @staticmethod
    def _load_cell_mean_std(img, local_center_point, i, img_idx):
        #local_center_point = local_center_point.astype(np.uint32)
        from_x = max(0, local_center_point[0] - HistogramDiffMinimization.__HALF_CELL_WIDTH)
        to_x = min(img.shape[1], local_center_point[0] + HistogramDiffMinimization.__HALF_CELL_WIDTH)
        from_y = max(0, local_center_point[1] - HistogramDiffMinimization.__HALF_CELL_HEIGHT)
        to_y = min(img.shape[0], local_center_point[1] + HistogramDiffMinimization.__HALF_CELL_HEIGHT)
        roi = img[from_y:to_y, from_x:to_x]
        #hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
        #return hist# / float(img.size)
        #cv2.imwrite('debug_entry{}_img_idx{}.png'.format(i, img_idx), roi)
        return np.array([np.mean(roi), np.std(roi)])


    # def dist_mean_std_scales(per_tile_scale, cells_pairs, means_stds):
    #     dist_means = []
    #     dist_stds = []
    #     for pair in cells_pairs:
    #         tile_idx1, [tile_cell_col1, tile_cell_row1], _ = pair[0]
    #         tile_idx2, [tile_cell_col2, tile_cell_row2], _ = pair[1]
    #         scale1 = per_tile_scale[tile_idx1]
    #         scale2 = per_tile_scale[tile_idx2]
    #         mean_std1 = means_stds[(tile_idx1, tile_cell_col1, tile_cell_row1)]
    #         mean_std2 = means_stds[(tile_idx2, tile_cell_col2, tile_cell_row2)]
    # 
    #         diff = np.abs(scale1 * mean_std1 - scale2 * mean_std2)
    #         # Compute the L1 norm distance of the two scaled histograms
    #         dist_means.append(diff[0])
    #         #dist_stds.append(diff[1])
    #         
    #     return dist_means

    @staticmethod
    def _optimize_hist_scales(imgs_len, cells_pairs, cached_means_stds, maxiter=100, epsilon=5, stepsize=0.8):
        # The initial histogram scale of each tile (1.0)
        per_tile_scale = np.ones((imgs_len), dtype=np.float)

        prev_med = np.inf
            
        for iter in range(maxiter):
            # compute the scaled mean and std for each pair of neighboring 
            dist_means = []
            dist_stds = []
            self_dist_means_ratios = defaultdict(list)
            for pair in cells_pairs:
                tile_idx1, _ = pair[0]
                tile_idx2, _ = pair[1]
                scale1 = per_tile_scale[tile_idx1]
                scale2 = per_tile_scale[tile_idx2]
                mean_std1 = cached_means_stds[pair[0]]
                mean_std2 = cached_means_stds[pair[1]]

                diff = np.abs(scale1 * mean_std1 - scale2 * mean_std2)
                # Compute the L1 norm distance of the two scaled histograms
                dist_means.append(diff[0])
                #dist_stds.append(diff[1])
                self_dist_means_ratios[tile_idx1].append((scale2 * mean_std2[0]) / (scale1 * mean_std1[0]))
                self_dist_means_ratios[tile_idx2].append((scale1 * mean_std1[0]) / (scale2 * mean_std2[0]))


            #dists = dist_mean_std_scales(per_tile_scale, cells_pairs, cached_means_stds)
            # compute the median
            med = np.median(dist_means)
            max_dist = np.max(dist_means)
            print("{}) med: {}, mean: {}, max: {}, stepsize: {}".format(iter, med, np.mean(dist_means), max_dist, stepsize))

                
            if med < prev_med:
                stepsize *= 1.1
                if stepsize > 1:
                    stepsize = 1
            else:
                stepsize *= 0.5

            if stepsize < 1e-30:
                #logger.report_event("Step size is small enough, finishing optimization", log_level=logging.INFO)
                print("Step size is small enough, finishing optimization")
                break

            # compute the per-tile gradient
            grads = np.zeros_like(per_tile_scale)
            
            for tile_idx in range(len(per_tile_scale)):
                avg_ratio = np.mean(self_dist_means_ratios[tile_idx])
                grads[tile_idx] = np.log(avg_ratio)
            
            # Update the per_tile_scale for the next iteration
            per_tile_scale += grads * stepsize
            #per_tile_scale[per_tile_scale > 1.5] = 1.5
            #per_tile_scale[per_tile_scale < 0.5] = 0.5
                    
            prev_med = med

        print("Final per_tile_scale average: {}, std: {}".format(np.mean(per_tile_scale), np.std(per_tile_scale)))
        return per_tile_scale


    @staticmethod
    def _load_image(img_fname, cached_images):
        if img_fname not in cached_images.keys():
            print("Loading {}".format(os.path.basename(img_fname)))
            img = cv2.imread(img_fname, 0)
            cached_images[img_fname] = img
        return cached_images[img_fname]

#     def compute_intensity_normalizer_optimization(self):
#         print("Parsing tilespec file: {}".format(self._optimized_ts_fname))
#         img_shape, fnames_transforms = self._parse_tilespec_fname()
# 
#         general_image_grid_centers = HistogramDiffMinimization._get_general_image_grid_cells_centers(img_shape)
# 
#         print("Creating section grid for: {}".format(self._optimized_ts_fname))
#         section_grid, overlapping_grid_cells_list = HistogramDiffMinimization._divide_section_images_to_grid_cells(fnames_transforms, general_image_grid_centers)
# 
#         # Now need to focus only on the "overlapping_grid_cells_list" inside the "section_grid"
#         # in order to find equalize their histograms
#         cached_means_stds = {} #(tile_idx, local_cell_center) -> mean and std of the rectangle around the local_cell_center of the tile in tile_idx
#         cells_pairs = [] # pairs of overlapping (tile_idx, local_cell_center) that land on the same section_grid cell
#         loaded_images = LRU(100) # maximum of 100 images will be in memory at any given time)
#         for i, overlapping_grid_cell in enumerate(overlapping_grid_cells_list):
#             #
#             #for img_idx, img_local_center in section_grid[overlapping_grid_cell]:
#             for tuple_entry in section_grid[overlapping_grid_cell]:
#                 img_idx, img_local_center = tuple_entry
#                 img = HistogramDiffMinimization._load_image(fnames_transforms[img_idx][0], loaded_images)
#                 cached_means_stds[tuple_entry] = HistogramDiffMinimization._load_cell_mean_std(img, img_local_center, i, img_idx)
# 
#             cells_pairs.extend(itertools.combinations(section_grid[overlapping_grid_cell], 2))
#             
#         print("Optimizing intensities scales for: {}".format(self._optimized_ts_fname))
#         per_tile_scale = HistogramDiffMinimization._optimize_hist_scales(len(fnames_transforms), cells_pairs, cached_means_stds)
# 
#         assert(len(fnames_transforms) == len(per_tile_scale))
#         self._scale_map = { tile_fname:tile_scale for (tile_fname, _), tile_scale in zip(fnames_transforms, per_tile_scale) }
#         print(self._scale_map)

    @staticmethod
    def _overlapping_bboxes(bbox1, bbox2):
        # Returns true if there is intersection between the bboxes or a full containment
        if (bbox1[0] < bbox2[1]) and (bbox1[1] > bbox2[0]) and \
           (bbox1[2] < bbox2[3]) and (bbox1[3] > bbox2[2]):
            return True
        return False

    @staticmethod
    def _find_overlaps_means(fnames_transforms_bboxes):
        loaded_images = LRU(100) # maximum of 100 images will be in memory at any given time)
        #cached_means_stds = {} #(tile_fname, local_cell_center) -> mean and std of the rectangle around the local_cell_center of the tile in tile_idx
        overlaps_means = {}
        for tile1_idx in range(len(fnames_transforms_bboxes)-1):
            for tile2_idx in range(tile1_idx + 1, len(fnames_transforms_bboxes)):
                tile1 = fnames_transforms_bboxes[tile1_idx]
                tile2 = fnames_transforms_bboxes[tile2_idx]
                bbox1 = tile1[2]
                bbox2 = tile2[2]
                overlap_bbox = [
                        max(bbox1[0], bbox2[0]),
                        min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]),
                        min(bbox1[3], bbox2[3])
                ]
                if overlap_bbox[1] - overlap_bbox[0] > 0.1 and overlap_bbox[3] - overlap_bbox[2] > 0.1:
                    # there's an overlap between the two tile
                    # find the original coordinates in each of the tiles (before transformations)
                    tile1_fname, tile1_transform, _ = tile1
                    tile2_fname, tile2_transform, _ = tile2
                    overlap_bbox_pts = np.array([
                        [overlap_bbox[0], overlap_bbox[2]],
                        [overlap_bbox[1], overlap_bbox[2]],
                        [overlap_bbox[1], overlap_bbox[3]],
                        [overlap_bbox[0], overlap_bbox[3]]
                    ])
                    orig_pts1 = tile1_transform.apply_inv(overlap_bbox_pts)
                    orig_pts2 = tile2_transform.apply_inv(overlap_bbox_pts)

                    min_xy1 = np.min(orig_pts1, axis=0).astype(int)
                    max_xy1 = np.max(orig_pts1, axis=0).astype(int)
                    min_xy2 = np.min(orig_pts2, axis=0).astype(int)
                    max_xy2 = np.max(orig_pts2, axis=0).astype(int)

                    img1 = HistogramDiffMinimization._load_image(tile1_fname, loaded_images)
                    roi1 = img1[max(min_xy1[1], 0):min(max_xy1[1], img1.shape[0]), max(min_xy1[0], 0):min(max_xy1[0], img1.shape[1])]
                    img2 = HistogramDiffMinimization._load_image(tile2_fname, loaded_images)
                    roi2 = img2[max(min_xy2[1], 0):min(max_xy2[1], img2.shape[0]), max(min_xy2[0], 0):min(max_xy2[0], img2.shape[1])]

                    #print("roi1 mean", np.mean(roi1))
                    #print("roi2 mean", np.mean(roi2))
                    #print("done")
                    overlaps_means[(tile1_idx, tile2_idx)] = (np.mean(roi1), np.mean(roi2))
            
        return overlaps_means

    @staticmethod
    def _optimize_hist_scales2(imgs_len, overlaps_means, maxiter=100, epsilon=5, stepsize=0.8):
        # The initial histogram scale of each tile (1.0)
        per_tile_scale = np.ones((imgs_len), dtype=np.float)

        prev_med = np.inf
            
        for iter in range(maxiter):
            # compute the scaled mean and std for each pair of neighboring 
            dist_means = []
            #dist_stds = []
            self_dist_means_ratios = defaultdict(list)
            for ((tile1_idx, tile2_idx), (tile1_mean, tile2_mean)) in overlaps_means.items():
                scale1 = per_tile_scale[tile1_idx]
                scale2 = per_tile_scale[tile2_idx]

                diff = np.abs(scale1 * tile1_mean - scale2 * tile2_mean)
                # Compute the L1 norm distance of the two scaled histograms
                dist_means.append(diff)
                #dist_stds.append(diff[1])
                self_dist_means_ratios[tile1_idx].append((scale2 * tile2_mean) / (scale1 * tile1_mean))
                self_dist_means_ratios[tile2_idx].append((scale1 * tile1_mean) / (scale2 * tile2_mean))


            #dists = dist_mean_std_scales(per_tile_scale, cells_pairs, cached_means_stds)
            # compute the median
            med = np.median(dist_means)
            max_dist = np.max(dist_means)
            print("{}) med: {}, mean: {}, max: {}, stepsize: {}".format(iter, med, np.mean(dist_means), max_dist, stepsize))

                
            if med < prev_med:
                stepsize *= 1.1
                if stepsize > 1:
                    stepsize = 1
            else:
                stepsize *= 0.5

            if stepsize < 1e-30:
                #logger.report_event("Step size is small enough, finishing optimization", log_level=logging.INFO)
                print("Step size is small enough, finishing optimization")
                break

            # compute the per-tile gradient
            grads = np.zeros_like(per_tile_scale)
            
            for tile_idx in range(len(per_tile_scale)):
                avg_ratio = np.mean(self_dist_means_ratios[tile_idx])
                grads[tile_idx] = np.log(avg_ratio)
            
            # Update the per_tile_scale for the next iteration
            per_tile_scale += grads * stepsize
            #per_tile_scale[per_tile_scale > 1.5] = 1.5
            #per_tile_scale[per_tile_scale < 0.5] = 0.5
                    
            prev_med = med

        print("Final per_tile_scale average: {}, std: {}".format(np.mean(per_tile_scale), np.std(per_tile_scale)))
        return per_tile_scale



    def compute_intensity_normalizer_optimization(self):
        print("Parsing tilespec file: {}".format(self._optimized_ts_fname))
        img_shape, fnames_transforms_bboxes = self._parse_tilespec_fname()

        print("finding overlapping tiles")
        overlaps_means = HistogramDiffMinimization._find_overlaps_means(fnames_transforms_bboxes)

        print("Optimizing intensities scales for: {}".format(self._optimized_ts_fname))
        per_tile_scale = HistogramDiffMinimization._optimize_hist_scales2(len(fnames_transforms_bboxes), overlaps_means)

        assert(len(fnames_transforms_bboxes) == len(per_tile_scale))
        self._scale_map = { tile_fname:tile_scale for (tile_fname, _, _), tile_scale in zip(fnames_transforms_bboxes, per_tile_scale) }
        print(self._scale_map)



if __name__ == '__main__':
    
    #in_json_fname = 'W08_Sec214_montaged_2tiles.json'
    #out_pkl_fname = 'W08_Sec214_montaged_2tiles_adjuster.pkl'
    in_json_fname = sys.argv[1]
    out_pkl_fname = sys.argv[2]

    diff_mini = HistogramDiffMinimization(in_json_fname)
    diff_mini.compute_intensity_normalizer_optimization()
    with open(out_pkl_fname, 'wb') as out_f:
        pickle.dump(diff_mini, out_f, pickle.HIGHEST_PROTOCOL)

    

