# Allows rendering a given tilespec
from .multiple_tiles_affine_renderer import MultipleTilesAffineRenderer, BlendType
from .single_tile_affine_renderer import SingleTileAffineRenderer
import json
import numpy as np
from common import trans_models


class TilespecAffineRenderer:

    def __init__(self, tilespec, blend_type=BlendType.LINEAR):
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
            raise Exception('Affine MULTI_BAND_SEAM not supported at the moment')
        else:
            raise Exception('Unknown blend type')

        self.single_tiles = [SingleTileAffineRenderer(
                                tile_ts["mipmapLevels"]["0"]["imageUrl"].replace("file://", ""), tile_ts["width"], tile_ts["height"], bbox=tile_ts["bbox"], transformation_models=[trans_models.Transforms.from_tilespec(modelspec) for modelspec in tile_ts["transforms"]], compute_mask=compute_mask, compute_distances=compute_distances)
                            for tile_ts in tilespec]
#         # Add the corresponding transformation
#         for tile_ts, tile in zip(tilespec, self.single_tiles):
#             for t in tile_ts["transforms"]:
#                 transform = models.Transforms.from_tilespec(t)
#                 tile.add_transformation(transform.get_matrix()[:2])

        self.multi_renderer = MultipleTilesAffineRenderer(self.single_tiles, blend_type=blend_type)
        

    def render(self):
        return self.multi_renderer.render()

    def crop(self, from_x, from_y, to_x, to_y):
        return self.multi_renderer.crop(from_x, from_y, to_x, to_y)

    def add_transformation(self, transform_matrix):
        """Adds a transformation to all tiles"""
        self.multi_renderer.add_transformation(transform_matrix[:2])

