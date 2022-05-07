# A class that takes a single image, applies affine transformations, and renders it
# (and possibly a pixel-mask to tell which pixels are coming from the image)
# The class will only load the image when the render function is called (lazy evaluation)
import cv2
import numpy as np
import math

class SingleTileAffineRenderer:
    

    def __init__(self, img_path, width, height, 
                 bbox=None,
                 transformation_models=[],
                 compute_mask=False,
                 compute_distances=True):
        self.img_path = img_path
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
        self.transform_matrix = np.eye(3)[:2]
        for model in transformation_models:
            self._add_transformation(model.get_matrix()[:2])
        self.update_img_transformed_corners_and_bbox()

        # Save for caching
        self.already_rendered = False

    def _add_transformation(self, transform_matrix):
        self.add_transformation(transform_matrix, update_corners=False)

    def add_transformation(self, transform_matrix, update_corners=True):
        assert(transform_matrix.shape == (2, 3))
        self.transform_matrix = np.dot(np.vstack((transform_matrix, [0., 0., 1.])), np.vstack((self.transform_matrix, [0., 0., 1.])))[:2]
        if update_corners:
            self.update_img_transformed_corners_and_bbox()

        # Remove any rendering
        self.already_rendered = False
        self.img = None

    def get_start_point(self):
        return (self.bbox[0], self.bbox[2])

    def get_bbox(self):
        return self.bbox

    def render(self):
        """Returns the rendered image (after transformation), and the start point of the image in global coordinates"""
        if self.already_rendered:
            return self.img, np.array([self.bbox[0], self.bbox[1]])

        img = cv2.imread(self.img_path, cv2.IMREAD_ANYDEPTH)
        adjusted_transform = self.transform_matrix[:2].copy()
        adjusted_transform[0][2] -= self.bbox[0]
        adjusted_transform[1][2] -= self.bbox[2]
        
        self.img = cv2.warpAffine(img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)
        self.already_rendered = True
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
                            ).astype(np.float32)
            self.weights = cv2.warpAffine(weights_img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)
        # Returns the transformed image and the start point
        return self.img, (self.bbox[0], self.bbox[2])

    def fetch_mask(self):
        assert(self.compute_mask)

        if not self.already_rendered:
            self.render()

        return self.mask, (self.bbox[0], self.bbox[2])

    def crop(self, from_x, from_y, to_x, to_y):
        """Returns the cropped image, its starting point, and the cropped mask (if the mask was computed).
           The given coordinates are specified using world coordinates."""
        # find the overlapping area of the given coordinates and the transformed tile
        overlapping_area = [max(from_x, self.bbox[0]), min(to_x, self.bbox[1]), max(from_y, self.bbox[2]), min(to_y, self.bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_mask = None
        # Make sure the image was rendered
        self.render()
        cropped_img = self.img[int(overlapping_area[2] - self.bbox[2]):int(overlapping_area[3] - self.bbox[2] + 1),
                               int(overlapping_area[0] - self.bbox[0]):int(overlapping_area[1] - self.bbox[0] + 1)]
        if self.compute_mask:
            cropped_mask = self.mask[int(overlapping_area[2] - self.bbox[2]):int(overlapping_area[3] - self.bbox[2] + 1),
                                     int(overlapping_area[0] - self.bbox[0]):int(overlapping_area[1] - self.bbox[0] + 1)]
        # Take only the parts that are overlapping
        return cropped_img, (overlapping_area[0], overlapping_area[2]), cropped_mask

    def crop_with_distances(self, from_x, from_y, to_x, to_y):
        """Returns the cropped image, its starting point, and the cropped image L1 distances of each pixel inside the image from the edge
           of the rendered image (if the mask was computed).
           The given coordinates are specified using world coordinates."""
        # find the overlapping area of the given coordinates and the transformed tile
        overlapping_area = [max(from_x, self.bbox[0]), min(to_x, self.bbox[1]), max(from_y, self.bbox[2]), min(to_y, self.bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_distances = None
        # Make sure the image was rendered
        self.render()
        cropped_img = self.img[int(overlapping_area[2] - self.bbox[2]):int(overlapping_area[3] - self.bbox[2] + 1),
                               int(overlapping_area[0] - self.bbox[0]):int(overlapping_area[1] - self.bbox[0] + 1)]

        if self.compute_distances:
            cropped_distances = self.weights[int(overlapping_area[2] - self.bbox[2]):int(overlapping_area[3] - self.bbox[2] + 1),
                                             int(overlapping_area[0] - self.bbox[0]):int(overlapping_area[1] - self.bbox[0] + 1)]
           
        # Take only the parts that are overlapping
        return cropped_img, (overlapping_area[0], overlapping_area[2]), cropped_distances



    # Helper methods (shouldn't be used from the outside)
    def update_img_transformed_corners_and_bbox(self):
        pts = np.array([[0., 0.], [self.width - 1, 0.], [self.width - 1, self.height - 1], [0., self.height - 1]])
        self.corners = np.dot(self.transform_matrix[:2,:2], pts.T).T + np.asarray(self.transform_matrix.T[2][:2]).reshape((1, 2))
        min_XY = np.min(self.corners, axis=0)
        max_XY = np.max(self.corners, axis=0)
        # Rounding to avoid float precision errors due to representation
        self.bbox = [int(math.floor(round(min_XY[0], 5))), int(math.ceil(round(max_XY[0], 5))), int(math.floor(round(min_XY[1], 5))), int(math.ceil(round(max_XY[1], 5)))]
        #self.bbox = [int(min_XY[0] + math.copysign(0.5, min_XY[0])), int(max_XY[0] + math.copysign(0.5, max_XY[1])), int(min_XY[1] + math.copysign(0.5, min_XY[1])), int(max_XY[1] + math.copysign(0.5, max_XY[1]))]
        self.shape = (self.bbox[1] - self.bbox[0] + 1, self.bbox[3] - self.bbox[2] + 1)




