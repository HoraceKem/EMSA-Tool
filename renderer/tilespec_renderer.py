# Allows rendering a given tilespec
from .multiple_tiles_renderer import MultipleTilesRenderer, BlendType
from .single_tile_renderer import SingleTileRenderer, SingleTileStaticRenderer
import numpy as np
from common import trans_models
import time


class TilespecRenderer:
    def __init__(self, tilespec, dtype=np.uint8, hist_adjuster=None, dynamic=True, blend_type=BlendType.MULTI_BAND_SEAM):
        st_time = time.time()
        if blend_type == BlendType.NO_BLENDING: # NO_BLENDING
            compute_mask = False
            compute_distances = False
        elif blend_type == BlendType.AVERAGING: # AVERAGING
            compute_mask = True
            compute_distances = False
        elif blend_type == BlendType.LINEAR: # LINEAR
            compute_mask = False
            compute_distances = True
        elif blend_type == BlendType.MULTI_BAND_SEAM: # MULTI_BAND_SEAM
            compute_mask = True
            compute_distances = False
        else:
            raise Exception('Unknown blend type')


        if dynamic:
#         self.single_tiles = [SingleTileRenderer(
#                                 tile_ts["mipmapLevels"]["0"]["imageUrl"].replace("file://", ""), tile_ts["width"], tile_ts["height"], compute_distances=True, hist_adjuster=hist_adjuster)
#                             for tile_ts in tilespec]
            self.single_tiles = [SingleTileRenderer(
                                    tile_ts["mipmapLevels"]["0"]["imageUrl"].replace("file://", ""), tile_ts["width"], tile_ts["height"], bbox=tile_ts["bbox"], transformation_models=[trans_models.Transforms.from_tilespec(modelspec) for modelspec in tile_ts["transforms"]], compute_mask=compute_mask, compute_distances=compute_distances, hist_adjuster=hist_adjuster)
                                for tile_ts in tilespec]
#         self.single_tiles = [SingleTileRenderer(
#                                 tile_ts["mipmapLevels"]["0"]["imageUrl"].replace("file://", ""), tile_ts["width"], tile_ts["height"], compute_mask=True, compute_distances=False, hist_adjuster=hist_adjuster)
#                             for tile_ts in tilespec]
        else: # non dynamic
            self.single_tiles = [SingleTileStaticRenderer(
                                    tile_ts["mipmapLevels"]["0"]["imageUrl"].replace("file://", ""), tile_ts["width"], tile_ts["height"], bbox=tile_ts["bbox"], transformation_models=[trans_models.Transforms.from_tilespec(modelspec) for modelspec in tile_ts["transforms"]], compute_mask=compute_mask, compute_distances=compute_distances, hist_adjuster=hist_adjuster)
                                for tile_ts in tilespec]

        print("single tile renderers setup time: {}".format(time.time() - st_time))
#         st_time = time.time()
#         # Add the corresponding transformation
#         for tile_ts, tile in zip(tilespec, self.single_tiles):
#             for t in tile_ts["transforms"]:
#                 model = models.Transforms.from_tilespec(t)
#                 tile.add_transformation(model)
# 
#         print("time: {}".format(time.time() - st_time))
        st_time = time.time()
        #self.multi_renderer = MultipleTilesRenderer(self.single_tiles, blend_type="LINEAR")
        #self.multi_renderer = MultipleTilesRenderer(self.single_tiles, blend_type="MULTI_BAND_SEAM")
        #self.multi_renderer = MultipleTilesRenderer(self.single_tiles, blend_type="AVERAGING")
        self.multi_renderer = MultipleTilesRenderer(self.single_tiles, blend_type=blend_type, dtype=dtype)
        print("multi tile renderer time: {}".format(time.time() - st_time))

       

    def render(self):
        return self.multi_renderer.render()

    def crop(self, from_x, from_y, to_x, to_y):
        return self.multi_renderer.crop(from_x, from_y, to_x, to_y)

    def add_transformation(self, model):
        """Adds a transformation to all tiles"""
        self.multi_renderer.add_transformation(model)

