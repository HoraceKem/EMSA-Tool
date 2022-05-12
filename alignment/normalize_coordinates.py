import os
import glob
from common.bounding_box import BoundingBox
import json
from common import utils

overall_args = utils.load_json_file('../arguments/overall_args.json')
log_controller = utils.LogController('alignment', os.path.join(overall_args["base"]["workspace"], 'log'),
                                     overall_args["base"]["running_mode"])


def add_transformation(input_file_path: str, output_file_path: str, transform: dict, deltas: list):
    """
    Add transformation to the tilespecs json file
    :param input_file_path:
    :param output_file_path:
    :param transform:
    :param deltas:
    :return:
    """
    # load the current json file
    with open(input_file_path, 'r') as f:
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

    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)
 

def normalize_coordinates(tile_file_path: str, output_folder_path: str):
    """
    Normalize coordinates
    :param tile_file_path:
    :param output_folder_path:
    :return:
    """
    # Get all the files that need to be normalized
    all_files = []

    log_controller.debug("Reading {}".format(tile_file_path))
    for file_or_dir in tile_file_path:
        if not os.path.exists(file_or_dir):
            log_controller.debug("{0} does not exist (file/directory), skipping".format(file_or_dir))
            continue

        if os.path.isdir(file_or_dir):
            actual_dir_files = glob.glob(os.path.join(file_or_dir, '*.json'))
            all_files.extend(actual_dir_files)
        else:
            all_files.append(file_or_dir)

    if len(all_files) == 0:
        log_controller.debug("No files for normalization found. Exiting.")
        return

    log_controller.debug("Normalizing coordinates of {0} files".format(len(all_files)))

    # Retrieve the bounding box of these files
    entire_image_bbox = None
    
    # merge the bounding boxes to a single bbox
    if len(all_files) > 0:
        entire_image_bbox = BoundingBox.read_bbox_grep(all_files[0])
        for f in all_files:
            entire_image_bbox.extend(BoundingBox.read_bbox_grep(f))
    
    log_controller.debug("Entire 3D image bounding box: {}".format(entire_image_bbox))

    # Set the translation transformation
    delta_x = entire_image_bbox.from_x
    delta_y = entire_image_bbox.from_y

    # TODO - use models to create the transformation
    transform = {
            "className": "mpicbg.trakem2.transform.TranslationModel2D",
            "dataString": "{0} {1}".format(-delta_x, -delta_y)
        }

    # Add the transformation to each tile in each tilespec
    for in_file in all_files:
        out_file = os.path.join(output_folder_path, os.path.basename(in_file))
        add_transformation(in_file, out_file, transform, [delta_x, delta_y])
