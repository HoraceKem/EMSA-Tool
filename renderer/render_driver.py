# @Time : 2020/11/18 上午10:59
# @Author : Horace.Kem
# @File: render_driver.py
# @Software: PyCharm

from renderer.tilespec_renderer import TilespecRenderer
from common import trans_models as models
from renderer.hist_matcher import HistMatcher
from renderer.multiple_tiles_renderer import BlendType
from common.bounding_box import BoundingBox
import cv2
import sys
import argparse
import numpy as np
import time
import json
import os
from common import utils
import glob
import multiprocessing as mp
import math


def pad_image(img, from_x, from_y, start_point):
    # Forked from rh-renderer(https://github.com/Rhoana/rh_renderer)
    """Pads the image (zeros) that starts from start_point (returned from the renderer), to (from_x, from_y)"""
    # Note that start_point is (y, x)
    if start_point[0] == from_y and start_point[1] == from_x:
        # Nothing to pad, return the image as is
        return img
    from_x = int(from_x)
    from_y = int(from_y)
    start_point = list(start_point)
    start_point[0] = int(start_point[0])
    start_point[1] = int(start_point[1])
    full_height_width = img.shape + np.array([start_point[1] - from_y, start_point[0] - from_x])
    full_img = np.zeros(full_height_width, dtype=img.dtype)
    full_img[start_point[1] - from_y:start_point[1] - from_y + img.shape[0],
    start_point[0] - from_x:start_point[0] - from_x + img.shape[1]] = img
    return full_img


def render_tilespec(tile_fname, output, scale, output_type, in_bbox, tile_size, invert_image,
                    empty_placeholder=False, reference_histogram_fname=None, from_to_cols_rows=None,
                    blendtype=BlendType.LINEAR):
    # Forked from rh-renderer(https://github.com/Rhoana/rh_renderer)
    """Renders a given tilespec.
       If the in_bbox to_x/to_y values are -1, uses the tilespecs to determine the output size.
       If tile_size is 0, the output will be a single image, otherwise multiple tiles will be created.
       output is either a single filename to save the output in (using the output_type),
       or a prefix for the tiles output, which will be of the form: {prefix}_tr%d-tc%d.{output_type}
       and the row (tr) and column (tc) values will be one-based."""
    if not os.path.exists(output):
        start_time = time.time()
        # Determine the output shape
        if in_bbox[1] == -1 or in_bbox[3] == -1:
            ts = utils.load_json_file(tile_fname)
            image_bbox = BoundingBox.read_bbox_from_ts(ts)
            image_bbox.from_x = max(image_bbox.from_x, in_bbox[0])
            image_bbox.from_y = max(image_bbox.from_y, in_bbox[2])
            if in_bbox[1] > 0:
                image_bbox.to_x = in_bbox[1]
            if in_bbox[3] > 0:
                image_bbox.to_y = in_bbox[3]
        else:
            image_bbox = BoundingBox.fromList(in_bbox)

        scaled_bbox = BoundingBox(
            int(math.floor(image_bbox.from_x * scale)),
            int(math.floor(image_bbox.to_x * scale)),
            int(math.floor(image_bbox.from_y * scale)),
            int(math.floor(image_bbox.to_y * scale))
        )
        # Set the post-scale out shape of the image
        out_shape = (scaled_bbox.to_x - scaled_bbox.from_x, scaled_bbox.to_y - scaled_bbox.from_y)
        print("Final out_shape for the image: {}".format(out_shape))

        reference_histogram = None
        if reference_histogram_fname is not None:
            reference_histogram = HistMatcher(histogram_fname=reference_histogram_fname)

        with open(tile_fname, 'r') as data:
            tilespec = json.load(data)
        renderer = TilespecRenderer(tilespec, hist_adjuster=reference_histogram, blend_type=blendtype)

        # Add the downsampling transformation
        downsample = models.AffineModel(np.array([
            [scale, 0., 0.],
            [0., scale, 0.],
            [0., 0., 1.]
        ]))
        renderer.add_transformation(downsample)

        if tile_size == 0:
            # no tiles, just render a single file
            out_fname = "{}.{}".format(os.path.splitext(output)[0], output_type)
            out_fname_empty = "{}_empty".format(out_fname)
            out_fname = output

            # Render the image
            img, start_point = renderer.crop(scaled_bbox.from_x, scaled_bbox.from_y, scaled_bbox.to_x - 1,
                                             scaled_bbox.to_y - 1)
            print("Rendered cropped and downsampled version")

            if empty_placeholder:
                if img is None or np.all(img == 0):
                    # create the empty file, and return
                    print("saving empty image {}".format(out_fname_empty))
                    open(out_fname_empty, 'a').close()
                    print("Rendering and saving empty file {} took {} seconds.".format(out_fname_empty,
                                                                                       time.time() - start_time))
                    return

            if img is None:
                # No actual image, set a blank image of the wanted size
                img = np.zeros((out_shape[1], out_shape[0]), dtype=np.uint8)
                start_point = (0, 0)

            print("Padding image")
            img = pad_image(img, scaled_bbox.from_x, scaled_bbox.from_y, start_point)

            if invert_image:
                print("inverting image")
                img = 255 - img
            print("saving image {}".format(out_fname))
            cv2.imwrite(out_fname, img)
        else:
            # Tile the image
            rows = int(math.ceil(out_shape[1] / float(tile_size)))
            cols = int(math.ceil(out_shape[0] / float(tile_size)))

            from_row = 0
            from_col = 0
            to_row = rows
            to_col = cols
            if from_to_cols_rows is not None:
                from_col, from_row, to_col, to_row = from_to_cols_rows

            # Iterate over each row and column and save the tile
            for cur_row in range(from_row, to_row):
                from_y = scaled_bbox.from_y + cur_row * tile_size
                to_y = min(scaled_bbox.from_y + (cur_row + 1) * tile_size, scaled_bbox.to_y)
                for cur_col in range(from_col, to_col):
                    tile_start_time = time.time()
                    out_fname = "{}_tr{}-tc{}.{}".format(output, str(cur_row + 1), str(cur_col + 1), output_type)
                    if os.path.exists(out_fname):
                        continue
                    out_fname_empty = "{}_empty".format(out_fname)
                    from_x = scaled_bbox.from_x + cur_col * tile_size
                    to_x = min(scaled_bbox.from_x + (cur_col + 1) * tile_size, scaled_bbox.to_x)

                    # Render the tile
                    img, start_point = renderer.crop(from_x, from_y, to_x - 1, to_y - 1)
                    print("Rendered cropped and downsampled version")

                    if empty_placeholder:
                        if img is None or np.all(img == 0):
                            # create the empty file, and return
                            print("saving empty image {}".format(out_fname_empty))
                            open(out_fname_empty, 'a').close()
                            continue

                    if img is None:
                        # No actual image, set a blank image of the wanted size
                        img = np.zeros((to_y - from_y, to_x - from_x), dtype=np.uint8)
                        start_point = (from_y, from_x)

                    print("Padding image")
                    img = pad_image(img, from_x, from_y, start_point)

                    if invert_image:
                        print("inverting image")
                        img = 255 - img

                    print("saving image {}".format(out_fname))
                    cv2.imwrite(out_fname, img)

                    print("single tile rendering and saving to {} took {} seconds.".format(out_fname,
                                                                                           time.time() - tile_start_time))
            # render_tilespec_mp(from_row, to_row, scaled_bbox, tile_size, from_col, to_col, output, output_type, renderer,
            #                    empty_placeholder, invert_image, processes_num=10)

        print("Rendering and saving {} took {} seconds.".format(tile_fname, time.time() - start_time))
