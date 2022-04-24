import glob
import os


def check_EM_type(section_folder_path: str) -> str:
    """
    Check if the data in section folder is singlebeam or multibeam
    :param section_folder_path:
    :type section_folder_path: str
    :return: a string of the data type
    """
    if 'S_' in section_folder_path:
        return 'singlebeam'
    elif 'S*R*' in section_folder_path:
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
    return img_file_paths_filtered


# def check_tiles_num(section_folder_path: str, EM_type: str) -> int:
#     """
#     Check the number of tiles in one section folder
#     :param section_folder_path:
#     :type section_folder_path: str
#     :param EM_type:
#     :type EM_type: str
#     :return: the number of tiles
#     """
#     tiles_num = 0
#     if EM_type == 'singlebeam':
#         tiles_num = len(ls_img_file_paths_singlebeam(section_folder_path))
#     elif EM_type == 'multibeam':
#
#         mfov_folder_paths = utils.ls_sub_folder_paths(section_folder_path)
#         for mfov_folder_path in mfov_folder_paths:
#             tiles_num += len(ls_img_file_paths_multibeam(mfov_folder_path))
#     else:
#         raise AssertionError('Wrong EM type.')
#     return tiles_num
