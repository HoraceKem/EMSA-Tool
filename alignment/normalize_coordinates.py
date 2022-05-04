import sys
import os
import glob
import argparse
from ..common.bounding_box import BoundingBox
import json

def add_transformation(in_file, out_file, transform, deltas):
    # load the current json file
    with open(in_file, 'r') as f:
        data = json.load(f)

    if deltas[0] != 0.0 and deltas[1] != 0.0:
        for tile in data:
            # Update the transformation
            if "transforms" not in tile.keys():
                tile["transforms"] = []
            tile["transforms"].append(transform)

            # Update the bbox
            if "bbox" in tile.keys():
                bbox = tile["bbox"]
                bbox_new = [bbox[0] - deltas[0], bbox[1] - deltas[0], bbox[2] - deltas[1], bbox[3] - deltas[1]]
                tile["bbox"] = bbox_new

    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)
 

def normalize_coordinates(tile_fnames_or_dir, output_dir):
    # Get all the files that need to be normalized
    all_files = []

    print "Reading {}".format(tile_fnames_or_dir)
    for file_or_dir in tile_fnames_or_dir:
        if not os.path.exists(file_or_dir):
            print "{0} does not exist (file/directory), skipping".format(file_or_dir)
            continue

        if os.path.isdir(file_or_dir):
            actual_dir_files = glob.glob(os.path.join(file_or_dir, '*.json'))
            all_files.extend(actual_dir_files)
        else:
            all_files.append(file_or_dir)

    if len(all_files) == 0:
        print "No files for normalization found. Exiting."
        return

    print "Normalizing coordinates of {0} files".format(len(all_files))

    # Retrieve the bounding box of these files
    entire_image_bbox = None
    
    # merge the bounding boxes to a single bbox
    if len(all_files) > 0:
        entire_image_bbox = BoundingBox.read_bbox_grep(all_files[0])
        for f in all_files:
            entire_image_bbox.extend(BoundingBox.read_bbox_grep(f))
    
    print "Entire 3D image bounding box: {}".format(entire_image_bbox)

    # Set the translation transformation
    deltaX = entire_image_bbox.from_x
    deltaY = entire_image_bbox.from_y

    # TODO - use models to create the transformation
    transform = {
            "className" : "mpicbg.trakem2.transform.TranslationModel2D",
            "dataString" : "{0} {1}".format(-deltaX, -deltaY)
        }

    # Add the transformation to each tile in each tilespec
    for in_file in all_files:
        out_file = os.path.join(output_dir, os.path.basename(in_file))
        add_transformation(in_file, out_file, transform, [deltaX, deltaY])



def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a list of tilespec file names, normalizes all the tilepecs to a single coordinate system starting from (0,0).')
    parser.add_argument('tile_files_or_dirs', metavar='tile_files_or_dirs', type=str, nargs='+',
                        help='a list of json files that need to be normalized or a directories of json files')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./after_norm)',
                        default='./after_norm')

    args = parser.parse_args()

    normalize_coordinates(args.tile_files_or_dirs, args.output_dir)

if __name__ == '__main__':
    main()

