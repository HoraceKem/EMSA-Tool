import os
import logging
import glob
import json
import termcolor
from pymage_size import get_image_size


class LogController(object):
    def __init__(self, module_name: str, log_folder_path: str):
        """
        Initialize a log controller according to the module name and log folder path
        :param module_name: the name of the module, in order to collect logs into different files
        :type module_name: str
        :param log_folder_path: the absolute path of the log folder
        :type log_folder_path: str
        """
        # Set the logger
        self.logger = logging.getLogger(__name__)
        log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

        # Set the StreamHandler to print logs in the console
        log_handler_console = logging.StreamHandler()
        log_handler_console.setFormatter(log_formatter)
        log_handler_console.setLevel(logging.DEBUG)

        # Set the FileHandler to output the logs to files
        create_dir(log_folder_path)
        log_file_basename = '%s.txt' % module_name
        log_file_path = os.path.join(log_folder_path, log_file_basename)
        log_handler_file = logging.FileHandler(log_file_path)
        log_handler_file.setFormatter(log_formatter)
        log_handler_file.setLevel(logging.INFO)

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
    basenames = os.listdir(folder_path)
    basenames.sort()
    absolute_paths = []
    for basename in basenames:
        absolute_path = os.path.join(folder_path, basename)
        absolute_paths.append(absolute_path)
    return absolute_paths


def ls_img_file_paths_singlebeam(folder_path: str) -> list:
    """
    List out the singlebeam image file paths in the folder and filter out the failed-taken images
    :param folder_path: the path to a folder
    :type folder_path: str
    :return: a sorted list of absolute paths of successfully-taken singlebeam images in the folder
    """
    img_file_paths = glob.glob(os.path.join(folder_path, 'Tile_r*-c*_*.tif'))
    img_file_paths_filtered = []
    for img_file_path in img_file_paths:
        fail_string = 'failed'
        if fail_string not in img_file_path:
            img_file_paths_filtered.append(img_file_path)
    return img_file_paths_filtered


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


def index_tilespec_multibeam(tilespecs: list) -> dict:
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


def index_tilespec_singlebeam(tilespecs: list) -> dict:
    """
    Given a section tilespecs returns a dictionary of [tile_index] to the tile's tilespec
    :param tilespecs: a list of tile specifications of singlebeam data
    :type tilespecs: list
    :return: a dictionary containing tilespecs
    """
    indexed_tilespecs = {}
    for tilespec in tilespecs:
        indexed_tilespecs[tilespec["tile_index"]] = tilespec
    return indexed_tilespecs


def parse_layer_range(layer_range: str) -> list:
    """
    Parse the layer range in string format and return a list containing all the layers (int)
    :param layer_range: a string containing multipart of layer ranges, e.g. '1-3, 8-9' == '1, 2, 3, 8, 9'
    :type layer_range: str
    :return: a sorted list of all the layers
    """
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
