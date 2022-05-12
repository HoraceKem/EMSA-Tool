import os
import csv
import re
import decimal
import utils

overall_args = utils.load_json_file('../arguments/overall_args.json')
log_controller = utils.LogController('common', os.path.join(overall_args["base"]["workspace"], 'log'),
                                     overall_args["base"]["running_mode"])


def img_base_name_decimal_key(tile_info: dict) -> decimal.Decimal:
    """
    Decimal key for sorting tile information
    :param tile_info: a dictionary containing tile information
    :type tile_info: dict
    :return: decimal key
    """
    base_name = tile_info["img_base_name"]
    return decimal.Decimal(''.join([c for c in base_name if c.isdigit()]))


def parse_init_coord_multibeam(section_folder_path: str) -> [list, list, list]:
    """
    Parsing the coordinates according to the file stored in every section folder (Specifically used for multibeam data)
    :param section_folder_path: the absolute path to the section folder
    :type section_folder_path: str
    :return:
    """
    log_controller.debug('Parsing initial coordinates to tilespec...')
    img_dict = {}
    img_file_paths = []
    x = []
    y = []
    coord_file_path = os.path.join(section_folder_path, "full_image_coordinates.txt")
    with open(coord_file_path, 'r') as csvfile:
        log_controller.debug('Loaded coordinates file {}'.format(coord_file_path))
        data_reader = csv.reader(csvfile, delimiter='\t')
        for row in data_reader:
            img_relative_path = row[0].replace('\\', os.sep)
            img_file_path = os.path.join(section_folder_path, img_relative_path)
            img_sec_mfov_beam = '_'.join(img_relative_path.split(os.sep)[-1].split('_')[:3])
            # Make sure that no duplicates appear
            if img_sec_mfov_beam not in img_dict.keys():
                img_file_paths.append(img_file_path)
                img_dict[img_sec_mfov_beam] = img_relative_path
                cur_x = float(row[1])
                cur_y = float(row[2])
                x.append(cur_x)
                y.append(cur_y)
            else:
                # Either the image is duplicated, or a newer version was taken,
                # so make sure that the newer version is used
                prev_img = img_dict[img_sec_mfov_beam]
                prev_img_date = prev_img.split(os.sep)[-1].split('_')[-1]
                curr_img_date = img_relative_path.split(os.sep)[-1].split('_')[-1]
                if curr_img_date > prev_img_date:
                    idx = img_file_paths.index(prev_img)
                    img_file_paths[idx] = img_file_path
                    img_dict[img_sec_mfov_beam] = img_relative_path

    return img_file_paths, x, y


def parse_init_coord_singlebeam(section_folder_path: str, overlap: float = 0.1) -> [list, list, list]:
    """
    Parsing the coordinates according to the row and col index (Specifically used for singlebeam data)
    :param section_folder_path: the absolute path to the section folder
    :type section_folder_path: str
    :param overlap: the initial overlapping rate between two tiles
    :return:
    """
    log_controller.debug('Parsing initial coordinates to tilespec...')
    log_controller.debug('Overlap rate: {}%'.format(overlap * 100))
    img_file_paths = utils.ls_img_file_paths_singlebeam(section_folder_path)
    x = []
    y = []
    for img_file_path in img_file_paths:
        img_dims = utils.read_img_dimensions(img_file_path)
        img_base_name = os.path.basename(img_file_path)
        filename_match = re.match('Tile_r([0-9]+)-c([0-9]+)_.*[.]tif+', img_base_name)
        row = int(filename_match.group(1))
        col = int(filename_match.group(2))
        offset_y = (row - 1) * img_dims[0] * (1.0 - overlap)
        offset_x = (col - 1) * img_dims[1] * (1.0 - overlap)

        x.append(int(offset_y))
        y.append(int(offset_x))
    return img_file_paths, x, y


def parse_section_singlebeam(section_folder_path: str) -> list:
    """
    Parse the information of one section for singlebeam
    :param section_folder_path: the absolute path to the section folder
    :return:
    """
    log_controller.debug('Parsing section in {}'.format(section_folder_path))
    img_file_paths, offset_x, offset_y = parse_init_coord_singlebeam(section_folder_path)
    section_info = []

    for i, img_file_path in enumerate(img_file_paths):
        img_dims = utils.read_img_dimensions(img_file_path)
        tile_info = {
            "img_file_path": img_file_path,
            "img_base_name": os.path.basename(img_file_path),
            "width": img_dims[1],
            "height": img_dims[0],
            "tx": offset_x[i],
            "ty": offset_y[i],
            "mfov": i,
            "tile_index": 1
        }
        section_info.append(tile_info)
    section_info.sort(key=img_base_name_decimal_key)
    log_controller.debug('Parsed {} tiles in section folder {}'.format(len(section_info), section_folder_path))
    return section_info


def parse_section_multibeam(section_folder_path: str) -> list:
    """
    Parse the information of one section for multibeam
    :param section_folder_path: the absolute path to the section folder
    :return:
    """
    log_controller.debug('Parsing section in %s'.format(section_folder_path))
    img_file_paths, offset_x, offset_y = parse_init_coord_multibeam(section_folder_path)
    section_info = []

    for i, img_file_path in enumerate(img_file_paths):
        img_dims = utils.read_img_dimensions(img_file_path)
        tile_info = {
            "img_file_path": img_file_path,
            "img_base_name": os.path.basename(img_file_path),
            "width": img_dims[1],
            "height": img_dims[0],
            "tx": offset_x[i],
            "ty": offset_y[i],
            "mfov": int(os.path.basename(img_file_path).split('_')[1]),
            "tile_index": int(os.path.basename(img_file_path).split('_')[2])
        }
        section_info.append(tile_info)
    section_info.sort(key=img_base_name_decimal_key)
    log_controller.debug('Parsed {} tiles in section folder {}'.format(len(section_info), section_folder_path))
    return section_info

