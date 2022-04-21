import os
import cv2
import sys
import glob
import json
import time
from pymage_size import get_image_size


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


def index_tilespec_multi(tile_specification):
    index = {}
    for ts in tile_specification:
        # mfov = str(ts["mfov"])
        mfov = ts["mfov"]
        if mfov not in index.keys():
            index[mfov] = {}
        # index[mfov][str(ts["tile_index"])] = ts
        index[mfov][ts["tile_index"]] = ts
    return index


def index_tilespec_single(tile_specification):
    index = {}
    for ts in tile_specification:
        index[ts["tile_index"]] = ts
    return index


def wait_after_file(filename, timeout_seconds):
    if timeout_seconds > 0:
        cur_time = time.time()
        mod_time = os.path.getmtime(filename)
        end_wait_time = mod_time + timeout_seconds
        while cur_time < end_wait_time:
            print("Waiting for file: {}".format(filename))
            cur_time = time.time()
            mod_time = os.path.getmtime(filename)
            end_wait_time = mod_time + timeout_seconds
            if cur_time < end_wait_time:
                time.sleep(end_wait_time - cur_time)


def parse_range(s):
    result = set()
    if s is not None and len(s) != 0:
        for part in s.split(','):
            x = part.split('-')
            result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)


def write_list_to_file(file_name, lst):
    with open(file_name, 'w') as out_file:
        for item in lst:
            out_file.write("%s\n" % item)


def read_layer_from_file(tiles_spec_fname):
    layer = None
    with open(tiles_spec_fname, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        if tile['layer'] is None:
            print("Error reading layer in one of the tiles in: {0}".format(tiles_spec_fname))
            sys.exit(1)
        if layer is None:
            layer = tile['layer']
        if layer != tile['layer']:
            print("Error when reading tiles from {0} found inconsistent layers numbers: {1} and {2}".format(
                tiles_spec_fname, layer, tile['layer']))
            sys.exit(1)
    if layer is None:
        print("Error reading layers file: {0}. No layers found.".format(tiles_spec_fname))
        sys.exit(1)
    return int(layer)


def save_h5(h5f, data, target):
    """
    :param h5f:
    :param data:
    :param target:
    :return:
    """
    shape_list = list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0] = 1
        dataset = h5f.create_dataset(target, data=data, maxshape=tuple(shape_list), chunks=True)
        return
    else:
        dataset = h5f[target]
    len_old = dataset.shape[0]
    len_new = len_old + data.shape[0]
    shape_list[0] = len_new
    dataset.resize(tuple(shape_list))
    dataset[len_old:len_new] = data


def disk_stat(path):
    disk = os.statvfs(path)
    percent = (disk.f_blocks - disk.f_bfree) * 100 / (disk.f_blocks - disk.f_bfree + disk.f_bavail)
    return percent


if __name__ == '__main__':
    filepaths = ls_absolute_paths('/Users/gehongyu/硕士资料/CECTSS')
    print(filepaths)
