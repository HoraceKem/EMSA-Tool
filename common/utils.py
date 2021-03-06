import os
import logging
import json
import glob
import math
import re

import termcolor
from pymage_size import get_image_size


class LogController(object):
    def __init__(self, module_name: str, file_name: str, log_folder_path: str):
        """
        Initialize a log controller according to the module name and log folder path
        :param module_name: the name of the module, in order to collect logs into different files
        :type module_name: str
        :param log_folder_path: the absolute path of the log folder
        :type log_folder_path: str
        """
        # Set the logger
        self.logger = logging.getLogger(file_name)
        self.logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(process)d] [%(name)s] '
                                          ' %(message)s', '%Y-%m-%d %H:%M:%S')

        # Set the StreamHandler to print logs in the console
        log_handler_console = logging.StreamHandler()
        log_handler_console.setFormatter(log_formatter)
        log_handler_console.setLevel(logging.INFO)

        # Set the FileHandler to output the logs to files
        create_dir(log_folder_path)
        log_file_basename = '%s.txt' % module_name
        log_file_path = os.path.join(log_folder_path, log_file_basename)
        log_handler_file = logging.FileHandler(log_file_path)
        log_handler_file.setFormatter(log_formatter)
        log_handler_file.setLevel(logging.DEBUG)

        self.logger.addHandler(log_handler_console)
        self.logger.addHandler(log_handler_file)

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)


def create_dir(folder_path: str):
    """
    Create a new folder recursively if not exist
    :param folder_path: the path to the new folder
    :type folder_path: str
    :return:
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def ls_absolute_paths(folder_path: str) -> list:
    """
    List out all the files' absolute path in the folder
    :param folder_path: the path to a folder
    :type folder_path: str
    :return: a sorted list of absolute paths in the folder
    """
    folder_path = os.path.abspath(folder_path)
    basenames = os.listdir(folder_path)
    basenames.sort()
    absolute_paths = []
    for basename in basenames:
        absolute_path = os.path.join(folder_path, basename)
        absolute_paths.append(absolute_path)
    return absolute_paths


def ls_sub_folder_paths(folder_path: str) -> list:
    """
    List out all the sub-folders' absolute path in the folder
    :param folder_path: the path to a folder
    :type folder_path: str
    :return: a sorted list of absolute paths in the folder
    """
    folder_path = os.path.abspath(folder_path)
    basenames = os.listdir(folder_path)
    basenames.sort()
    sub_folder_paths = []
    for basename in basenames:
        absolute_path = os.path.join(folder_path, basename)
        if os.path.isdir(absolute_path):
            sub_folder_paths.append(absolute_path)
    return sub_folder_paths


def get_section_folder_paths(folder_path: str, folder_depth: int) -> list:
    """
    Recursively find the section folder paths from a given folder
    :param folder_path:
    :param folder_depth:
    :return:
    """
    if folder_depth == 0:
        return [folder_path]
    else:
        res = []
        sub_folder_paths = ls_sub_folder_paths(folder_path)
        for sub_folder_path in sub_folder_paths:
            res.extend(get_section_folder_paths(sub_folder_path, folder_depth - 1))
    return res


def read_img_dimensions(img_file_path: str) -> tuple:
    """
    Read the image dimensions(height, width) without loading it into memory.
    This function is powered by the 'pymage_size' module
    :param img_file_path: the path to the image
    :type img_file_path: str
    :return: a tuple of dimensions
    """
    image_format = get_image_size(img_file_path)
    width, height = image_format.get_dimensions()
    dims = (height, width)
    return dims


def load_json_file(json_file_path: str):
    """
    Load the contents in json file
    :param json_file_path: the path to the json file
    :type json_file_path: str
    :return: the contents, which can be a dictionary, a list, etc.
    """
    with open(json_file_path, 'r') as f:
        json_contents = json.load(f)
    return json_contents


def save_json_file(output_json_file_path: str, json_contents):
    """
    Save the contents into json file
    :param output_json_file_path: 
    :param json_contents: 
    :return: 
    """
    with open(output_json_file_path, 'w') as f:
        json.dump(json_contents, f, sort_keys=False, indent=4)


def index_tilespec(tilespecs: list) -> dict:
    """
    Given a section tilespecs returns a dictionary of [mfov][tile_index] to the tile's tilespec
    :param tilespecs: a list of tile specifications of multibeam data
    :type tilespecs: list
    :return: a dictionary containing tilespecs
    """
    indexed_tilespecs = {}
    for tilespec in tilespecs:
        mfov = tilespec["mfov"]
        if mfov not in indexed_tilespecs.keys():
            indexed_tilespecs[mfov] = {}
        indexed_tilespecs[mfov][tilespec["tile_index"]] = tilespec
    return indexed_tilespecs


def parse_layer_range(layer_range: str) -> list:
    """
    Parse the layer range in string format and return a list containing all the layers (int)
    :param layer_range: a string containing multipart of layer ranges, e.g. '1-3, 8-9' == '1, 2, 3, 8, 9'
    :type layer_range: str
    :return: a sorted list of all the layers
    """
    if layer_range == 'None':
        return []
    layers = set()
    if layer_range is not None and len(layer_range) != 0:
        for part in layer_range.split(','):
            layer_num = part.split('-')
            layers.update(set(range(int(layer_num[0]), int(layer_num[-1]) + 1)))
    return sorted(layers)


def save_list(file_path: str, to_be_saved_list: list):
    """
    Save a list to text file
    :param file_path:
    :type file_path: str
    :param to_be_saved_list:
    :type to_be_saved_list: list
    :return:
    """
    with open(file_path, 'w') as f:
        for item in to_be_saved_list:
            f.write("%s\n" % item)
        f.close()


def get_occupied_space_pct(absolute_path: str) -> float:
    """
    Get the occupied storage space in the percent format. Used to spy on the storage device's status
    :param absolute_path:
    :type absolute_path: str
    :return: a float percent of occupied space
    """
    disk = os.statvfs(absolute_path)
    percent = (disk.f_blocks - disk.f_bfree) * 100 / (disk.f_blocks - disk.f_bfree + disk.f_bavail)
    return percent


def to_red(content: str) -> str:
    """
    Change the text to red
    :param content:
    :return:
    """
    return termcolor.colored(content, "red", attrs=["bold"])


def to_green(content: str) -> str:
    """
    Change the text to green
    :param content:
    :return:
    """
    return termcolor.colored(content, "green", attrs=["bold"])


def to_blue(content: str) -> str:
    """
    Change the text to blue
    :param content:
    :return:
    """
    return termcolor.colored(content, "blue", attrs=["bold"])


def to_cyan(content: str) -> str:
    """
    Change the text to cyan
    :param content:
    :return:
    """
    return termcolor.colored(content, "cyan", attrs=["bold"])


def to_yellow(content: str) -> str:
    """
    Change the text to yellow
    :param content:
    :return:
    """
    return termcolor.colored(content, "yellow", attrs=["bold"])


def to_magenta(content: str) -> str:
    """
    Change the text to magenta
    :param content:
    :return:
    """
    return termcolor.colored(content, "magenta", attrs=["bold"])


def check_EM_type(section_folder_path: str) -> str:
    """
    Check if the data in section folder is singlebeam or multibeam
    :param section_folder_path:
    :type section_folder_path: str
    :return: a string of the data type
    """
    if 'S_' in section_folder_path:
        return 'singlebeam'
    elif re.match('([0-9]+)_[S]([0-9]+)[R]([0-9]+)', os.path.basename(section_folder_path)):
        return 'multibeam'
    return 'unknown'


def ls_img_file_paths_singlebeam(section_folder_path: str, file_ext: str = 'tif') -> list:
    """
    List out the singlebeam image file paths in the folder and filter out the failed-taken images
    :param section_folder_path: the path to a folder
    :type section_folder_path: str
    :param file_ext: file extension (not including dot)
    :type file_ext: str
    :return: a sorted list of absolute paths of successfully-taken singlebeam images in the section folder
    """
    img_file_paths = glob.glob(os.path.join(section_folder_path, 'Tile_r*-c*_*' + '.' + file_ext))
    img_file_paths_filtered = []
    for img_file_path in img_file_paths:
        fail_string = 'failed'
        if fail_string not in img_file_path:
            img_file_paths_filtered.append(img_file_path)
    img_file_paths_filtered.sort()
    return img_file_paths_filtered


def ls_img_file_paths_multibeam(mfov_folder_path: str, file_ext: str = 'bmp') -> list:
    """
    List out the multibeam image file paths in the folder and filter out the thumbnail images
    :param mfov_folder_path: the path to a folder
    :type mfov_folder_path: str
    :param file_ext: file extension (not including dot)
    :type file_ext: str
    :return: a sorted list of absolute paths of full-resolution images in the mfov folder
    """
    img_file_paths = glob.glob(os.path.join(mfov_folder_path, '*_*_*' + '.' + file_ext))
    img_file_paths_filtered = []
    for img_file_path in img_file_paths:
        thumbnail_string = 'thumbnail'
        if thumbnail_string not in img_file_path:
            img_file_paths_filtered.append(img_file_path)
    img_file_paths_filtered.sort()
    return img_file_paths_filtered


def check_tiles_num(section_folder_path: str, EM_type: str) -> int:
    """
    Check the number of tiles in one section folder
    :param section_folder_path:
    :type section_folder_path: str
    :param EM_type:
    :type EM_type: str
    :return: the number of tiles
    """
    tiles_num = 0
    if EM_type == 'singlebeam':
        tiles_num = len(ls_img_file_paths_singlebeam(section_folder_path))
    elif EM_type == 'multibeam':
        mfov_folder_paths = ls_sub_folder_paths(section_folder_path)
        for mfov_folder_path in mfov_folder_paths:
            tiles_num += len(ls_img_file_paths_multibeam(mfov_folder_path))
    else:
        raise AssertionError('Wrong EM type.')
    return tiles_num


def generate_hexagonal_grid(bbox: list, spacing: int) -> list:
    """
    Generates a hexagonal grid inside a given bounding-box with a given spacing between the vertices
    :param bbox:
    :param spacing:
    :return:
    """
    hex_height = spacing
    hex_width = math.sqrt(3) * spacing / 2
    vertical_spacing = 0.75 * hex_height
    horizontal_spacing = hex_width
    size_x = int((bbox[1] - bbox[0]) / horizontal_spacing) + 2
    size_y = int((bbox[3] - bbox[2]) / vertical_spacing) + 2
    if size_y % 2 == 0:
        size_y += 1
    pts = []
    for i in range(-2, size_x):
        for j in range(-2, size_y):
            x_loc = i * spacing
            y_loc = j * spacing
            if j % 2 == 1:
                x_loc += spacing * 0.5
            if (j % 2 == 1) and (i == size_x - 1):
                continue
            pts.append([int(x_loc + bbox[0]), int(y_loc + bbox[2])])
    return pts


def read_layer_from_tilespecs_file(tilespecs_file_path: str) -> int:
    """
    Read the layer from a tilespecs json file.
    One json file should have one and only one layer number
    :param tilespecs_file_path:
    :return:
    """
    layer = None
    with open(tilespecs_file_path, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        if tile['layer'] is None:
            raise AssertionError("Error reading layer in one of the tiles in: {0}".format(tilespecs_file_path))
        if layer is None:
            layer = tile['layer']
        if layer != tile['layer']:
            raise AssertionError("Error when reading tiles from {0} found inconsistent layers "
                                 "numbers: {1} and {2}".format(tilespecs_file_path, layer, tile['layer']))
    if layer is None:
        raise AssertionError("Error reading layers file: {0}. No layers found.".format(tilespecs_file_path))
    return int(layer)


def write_list_to_file(file_path: str, list_data: list):
    """
    Write a list into a file
    :param file_path:
    :param list_data:
    :return:
    """
    with open(file_path, 'w') as out_file:
        for item in list_data:
            out_file.write("%s\n" % item)


class suppress_stdout_stderr(object):
    """
    [3rd party]
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
