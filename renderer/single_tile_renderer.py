# A class that takes a single image, applies transformations (both affine and non-affine), and renders it
# (and possibly a pixel-mask to tell which pixels are coming from the image).
# Assumption: there is only one non-affine transformation. TODO - get rid of this assumption
# The class will only load the image when the render function is called (lazy evaluation).
# Consecutive affine transformations will be condensed into a single transformation
import cv2
import numpy as np
import math
from common.trans_models import AffineModel
import scipy.interpolate as spint
import time
from common import utils
import os
overall_args = utils.load_json_file('arguments/overall_args.json')
log_controller = utils.LogController('render', 'single_tile_renderer',
                                     os.path.join(overall_args["base"]["workspace"], 'log'))


class SingleTileRendererBase(object):
    def __init__(self, width, height, 
                 bbox=None,
                 transformation_models=[],
                 compute_mask=False, 
                 compute_distances=True):
        self.img = None
        self.width = width
        self.height = height
        self.compute_mask = compute_mask
        self.mask = None
        self.compute_distances = compute_distances
        self.weights = None
        if bbox is None:
            self.bbox = [0, width - 1, 0, height - 1]
            self.shape = (width, height)
        else:
            self.bbox = np.around(bbox).astype(int)
            self.shape = (self.bbox[1] - self.bbox[0] + 1, self.bbox[3] - self.bbox[2] + 1)

        self.start_point = (self.bbox[0], self.bbox[2])
        # If only affine is used then this is always (bbox[0], bbox[2]), with non-affine it might be different

        # Starting with a single identity affine transformation
        self.pre_non_affine_transform = np.eye(3)[:2]
        self.non_affine_transform = None
        self.post_non_affine_transform = np.eye(3)[:2]
        for model in transformation_models:
            self._add_transformation(model)

        # Save for caching
        self.already_rendered = False

    def _add_transformation(self, model):
        if model.is_affine():
            new_model_matrix = model.get_matrix()
            # Need to add the transformation either to the pre_non_affine or the post_non_affine
            if self.non_affine_transform is None:
                cur_transformation = self.pre_non_affine_transform
            else:
                cur_transformation = self.post_non_affine_transform

            # Compute the new transformation (multiply from the left)
            new_transformation = np.dot(new_model_matrix, np.vstack((cur_transformation, [0., 0., 1.])))[:2]

            if self.non_affine_transform is None:
                self.pre_non_affine_transform = new_transformation
            else:
                self.post_non_affine_transform = new_transformation
        else:
            # Non-affine transformation
            self.non_affine_transform = model

        # Remove any rendering
        self.already_rendered = False
        self.img = None

    def get_bbox(self):
        return self.bbox

    def load(self):
        raise NotImplementedError("Please implement load in a derived class")
    
    def render(self):
        """Returns the rendered image (after transformation), and the start point of the image in global coordinates"""
        if self.already_rendered:
            return self.img, self.start_point

        st_time = time.time()
        img = self.load()
        log_controller.debug("loading image time: {}".format(time.time() - st_time))
        self.start_point = np.array([self.bbox[0], self.bbox[2]])  # may be different for non-affine result

        if self.non_affine_transform is None:
            # If there wasn't a non-affine transformation, we only need to apply an affine transformation
            adjusted_transform = self.pre_non_affine_transform[:2].copy()
            adjusted_transform[0][2] -= self.bbox[0]
            adjusted_transform[1][2] -= self.bbox[2]

            self.img = cv2.warpAffine(img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)
            if self.compute_mask:
                mask_img = np.ones(img.shape)
                self.mask = cv2.warpAffine(mask_img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)
                self.mask[self.mask > 0] = 1
                self.mask = self.mask.astype(np.uint8)
            if self.compute_distances:
                # The initial weights for each pixel is the minimum from the image boundary
                grid = np.mgrid[0:self.height, 0:self.width]
                weights_img = np.minimum(
                                    np.minimum(grid[0], self.height - 1 - grid[0]),
                                    np.minimum(grid[1], self.width - 1 - grid[1])
                                ).astype(np.float32) + .5
                self.weights = cv2.warpAffine(weights_img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)

        else:
            # Apply a reverse pre affine transformation on the source points of the non-affine transformation,
            # and a post affine transformation on the destination points
            src_points, dest_points = self.non_affine_transform.get_point_map()
            inverted_pre = np.linalg.inv(np.vstack([self.pre_non_affine_transform, [0., 0., 1.]]))[:2]
            src_points = np.dot(inverted_pre[:2, :2], src_points.T).T + inverted_pre[:, 2].reshape((1, 2))
            dest_points = np.dot(self.post_non_affine_transform[:2, :2], dest_points.T).T + \
                          self.post_non_affine_transform[:, 2].reshape((1, 2))

            # Move the destination points to start at (0, 0) --> less rendering
            dest_points = dest_points - np.array([self.bbox[0], self.bbox[2]])

            # Set the target grid using the shape
            out_grid_x, out_grid_y = np.mgrid[0:self.shape[0], 0:self.shape[1]]

            # TODO - is there a way to further restrict the target grid size, and speed up the interpolation?
            # Use griddata to interpolate all the destination points
            out_grid_z = spint.griddata(dest_points, src_points, (out_grid_x, out_grid_y),
                                        method='linear', fill_value=-1.)

            map_x = np.append([], [ar[:, 0] for ar in out_grid_z]).reshape(self.shape[0],
                                                                           self.shape[1]).astype('float32')
            map_y = np.append([], [ar[:, 1] for ar in out_grid_z]).reshape(self.shape[0],
                                                                           self.shape[1]).astype('float32')
            # find all rows and columns that are mapped before or after the boundaries of
            # the source image, and remove them
            map_valid_cells = np.where((map_x >= 0.) & (map_x < float(self.width)) &
                                       (map_y >= 0.) & (map_y < float(self.height)))
            min_col_row = np.min(map_valid_cells, axis=1)
            max_col_row = np.max(map_valid_cells, axis=1)
            map_x = map_x[min_col_row[0]:max_col_row[0], min_col_row[1]:max_col_row[1]]
            map_y = map_y[min_col_row[0]:max_col_row[0], min_col_row[1]:max_col_row[1]]

            # remap the source points to the destination points
            self.img = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC).T
            self.start_point = self.start_point + min_col_row
            # Add mask and weights computation
            if self.compute_mask:
                mask_img = np.ones(img.shape)
                self.mask = cv2.remap(mask_img, map_x, map_y, cv2.INTER_CUBIC).T
                self.mask[self.mask > 0] = 1
                self.mask = self.mask.astype(np.uint8)
            if self.compute_distances:
                # The initial weights for each pixel is the minimum from the image boundary
                grid = np.mgrid[0:self.height, 0:self.width]
                weights_img = np.minimum(
                                    np.minimum(grid[0], self.height - 1 - grid[0]),
                                    np.minimum(grid[1], self.width - 1 - grid[1])
                                ).astype(np.float32) + .5
                self.weights = cv2.remap(weights_img, map_x, map_y, cv2.INTER_CUBIC).T
                self.weights[self.weights < 0] = 0

        self.already_rendered = True
        return self.img, self.start_point

    def fetch_mask(self):
        assert self.compute_mask
        if not self.already_rendered:
            self.render()

        return self.mask, (self.bbox[0], self.bbox[2])

    def crop(self, from_x, from_y, to_x, to_y):
        """Returns the cropped image, its starting point, and the cropped mask (if the mask was computed).
           The given coordinates are specified using world coordinates."""
        log_controller.debug("!!crop called with from_xy: {},{} to_xy: {},{}".format(from_x, from_y, to_x, to_y))
        # find the overlapping area of the given coordinates and the transformed tile
        overlapping_area = [max(from_x, self.bbox[0]), min(to_x, self.bbox[1]),
                            max(from_y, self.bbox[2]), min(to_y, self.bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            log_controller.debug("overlapping area is empty: width:{}, height:{}".format(overlapping_width,
                                                                                         overlapping_height))
            return None, None, None

        cropped_mask = None
        # Make sure the image was rendered
        self.render()
        # Check with the actual image bounding box (maybe different because of the non-affine transformation)
        actual_bbox = [self.start_point[0], self.start_point[0] + self.img.shape[1],
                       self.start_point[1], self.start_point[1] + self.img.shape[0]]
        overlapping_area = [max(from_x, actual_bbox[0]), min(to_x, actual_bbox[1]),
                            max(from_y, actual_bbox[2]), min(to_y, actual_bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_img = self.img[int(overlapping_area[2] - actual_bbox[2]):int(overlapping_area[3] - actual_bbox[2] + 1),
                               int(overlapping_area[0] - actual_bbox[0]):int(overlapping_area[1] - actual_bbox[0] + 1)]
        if self.compute_mask:
            cropped_mask = self.mask[int(overlapping_area[2] - actual_bbox[2]):
                                     int(overlapping_area[3] - actual_bbox[2] + 1),
                                     int(overlapping_area[0] - actual_bbox[0]):
                                     int(overlapping_area[1] - actual_bbox[0] + 1)]
        # Take only the parts that are overlapping
        return cropped_img, (overlapping_area[0], overlapping_area[2]), cropped_mask

    def crop_with_distances(self, from_x, from_y, to_x, to_y):
        """Returns the cropped image, its starting point, and the cropped image L1 distances of each pixel inside the
        image from the edge of the rendered image (if the mask was computed).
        The given coordinates are specified using world coordinates."""
        # find the overlapping area of the given coordinates and the transformed tile
        overlapping_area = [max(from_x, self.bbox[0]), min(to_x, self.bbox[1]),
                            max(from_y, self.bbox[2]), min(to_y, self.bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_distances = None
        # Make sure the image was rendered
        self.render()
        # Check with the actual image bounding box (maybe different because of the non-affine transformation)
        actual_bbox = [self.start_point[0], self.start_point[0] + self.img.shape[1],
                       self.start_point[1], self.start_point[1] + self.img.shape[0]]
        overlapping_area = [max(from_x, actual_bbox[0]), min(to_x, actual_bbox[1]),
                            max(from_y, actual_bbox[2]), min(to_y, actual_bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_img = self.img[int(overlapping_area[2] - actual_bbox[2]):int(overlapping_area[3] - actual_bbox[2] + 1),
                               int(overlapping_area[0] - actual_bbox[0]):int(overlapping_area[1] - actual_bbox[0] + 1)]
        if self.compute_distances:
            cropped_distances = self.weights[int(overlapping_area[2] - actual_bbox[2]):
                                             int(overlapping_area[3] - actual_bbox[2] + 1),
                                             int(overlapping_area[0] - actual_bbox[0]):
                                             int(overlapping_area[1] - actual_bbox[0] + 1)]
           
        # Take only the parts that are overlapping
        return cropped_img, (overlapping_area[0], overlapping_area[2]), cropped_distances


class SingleTileDynamicRendererBase(SingleTileRendererBase):
    def __init__(self, width, height, 
                 bbox=None,
                 transformation_models=[],
                 compute_mask=False, 
                 compute_distances=True):
        super(SingleTileDynamicRendererBase, self).__init__(
            width, height, bbox, transformation_models, compute_mask, compute_distances)
        # Store the pixel locations (x,y) of the surrounding polygon of the image
        self.surrounding_polygon = np.array([[0., 0.], [width - 1., 0.], [width - 1., height - 1.], [0., height - 1.]])

        # update the surrounding polygon according to the model
        for model in transformation_models:
            self._update_surrounding_polygon(model)

    def add_transformation(self, model):
        # Call the private add transformation method in the parent
        super(SingleTileDynamicRendererBase, self)._add_transformation(model)

        # update the surrounding polygon according to the model
        self._update_surrounding_polygon(model)

        # Update bbox and shape according to the new borders
        self.bbox, self.shape = compute_bbox_and_shape(self.surrounding_polygon)

    def _update_surrounding_polygon(self, model):
        # Update the surrounding_polygon according to the new model
        if model.is_affine():
            self.surrounding_polygon = model.apply(self.surrounding_polygon)
        else:
            # TODO - need to see if this returns a sufficient bounding box for the reverse transformation
            # Find the new surrounding polygon locations
            # using a forward transformation from the boundaries of the source image to the destination
            boundary1 = np.array([[float(p), 0.] for p in np.arange(self.width)])
            boundary2 = np.array([[float(p), float(self.height - 1)] for p in np.arange(self.width)])
            boundary3 = np.array([[0., float(p)] for p in np.arange(self.height)])
            boundary4 = np.array([[float(self.width - 1), float(p)] for p in np.arange(self.height)])
            boundaries = np.concatenate((boundary1, boundary2, boundary3, boundary4))
            boundaries = np.dot(self.pre_non_affine_transform[:2, :2], boundaries.T).T + \
                         self.pre_non_affine_transform[:, 2].reshape((1, 2))
            self.surrounding_polygon = model.apply(boundaries)


class SingleTileStaticRenderer(SingleTileRendererBase):
    """
    Implementation of SingleTileRendererBase with file path for static (no further transformations) images
    """
    def __init__(self, img_path, width, height, 
                 bbox=None,
                 transformation_models=[],
                 compute_mask=False, 
                 compute_distances=True,
                 hist_adjuster=None):
        super(SingleTileStaticRenderer, self).__init__(
            width, height, bbox, transformation_models, compute_mask, compute_distances)
        self.img_path = img_path
        self.hist_adjuster = hist_adjuster
        
    def load(self):
        img = cv2.imread(self.img_path, cv2.IMREAD_ANYDEPTH)
        # Normalize the histogram if needed
        if self.hist_adjuster is not None:
            img = self.hist_adjuster.adjust_histogram(self.img_path, img)
        return img


class SingleTileRenderer(SingleTileDynamicRendererBase):
    """
    Implementation of SingleTileRendererBase with file path for dynamic (new transformations can be applied) images
    """
    def __init__(self, img_path, width, height, 
                 bbox=None,
                 transformation_models=[],
                 compute_mask=False, 
                 compute_distances=True,
                 hist_adjuster=None):
        super(SingleTileRenderer, self).__init__(
            width, height, bbox, transformation_models, compute_mask, compute_distances)
        self.img_path = img_path
        self.hist_adjuster = hist_adjuster
        
    def load(self):
        img = cv2.imread(self.img_path, cv2.IMREAD_ANYDEPTH)
        # Normalize the histogram if needed
        if self.hist_adjuster is not None:
            img = self.hist_adjuster.adjust_histogram(self.img_path, img)

        return img


class AlphaTileRenderer(SingleTileDynamicRendererBase):
    """
    An alpha channel for a pre-existing single tile
    """
    def __init__(self, other_renderer):
        """
        Initialize with another renderer
        :param other_renderer: A renderer derived from SingleTileRendererBase
        """
        super(AlphaTileRenderer, self).__init__(
            other_renderer.width, other_renderer.height, None, [], False, False)
        pre, post = [
            AffineModel(np.vstack([transform, [0, 0, 1]])
                        if transform.shape[0] == 2
                        else transform) 
            for transform in 
            [other_renderer.pre_non_affine_transform,
             other_renderer.post_non_affine_transform]]
        self.add_transformation(pre)             
        if other_renderer.non_affine_transform is not None:
            self.add_transformation(other_renderer.non_affine_transform)
            self.add_transformation(post)
    
    def load(self):
        return np.ones((self.height, self.width), np.float32)


# Helper methods (shouldn't be used from the outside)
def compute_bbox_and_shape(polygon):
    # find the new bounding box
    min_XY = np.min(polygon, axis=0)
    max_XY = np.max(polygon, axis=0)
    # Rounding to avoid float precision errors due to representation
    new_bbox = [int(math.floor(round(min_XY[0], 5))), int(math.ceil(round(max_XY[0], 5))),
                int(math.floor(round(min_XY[1], 5))), int(math.ceil(round(max_XY[1], 5)))]
    new_shape = (new_bbox[1] - new_bbox[0] + 1, new_bbox[3] - new_bbox[2] + 1)
    return new_bbox, new_shape
