import cv2
import numpy as np
import h5py
import os
import common.utils as utils
import common.extract_features as extract_features

overall_args = utils.load_json_file('../arguments/overall_args.json')
utils.create_dir(overall_args["base"]["workspace"])
log_controller = utils.LogController('stitching', os.path.join(overall_args["base"]["workspace"], 'log'))


def compute_and_save_tile_features(tilespec: dict, output_h5_file_path: str, features_type: str, features_args: dict):
    """
    Compute and save features as H5PY file according to the tilespec
    :param features_args:
    :param tilespec:
    :param output_h5_file_path:
    :param features_type:
    :return:
    """
    if features_type not in extract_features.__dict__:
        log_controller.error('Features type not defined. Please refer to common/extract_features.py')
        raise TypeError('features type')
    img_file_path = tilespec["mipmapLevels"]["0"]["imageUrl"]
    img_gray = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
    log_controller.debug("Computing {} features for image: {}".format(features_type, os.path.basename(img_file_path)))

    pts, des = extract_features.__dict__[features_type](img_gray, features_args)
    if des is None:
        pts = []
        des = []
        log_controller.warning('No feature point detected for {}.'.format(img_file_path))

    log_controller.debug("Saving {} {} features at: {}".format(len(pts), features_type, output_h5_file_path))
    with h5py.File(output_h5_file_path, 'w') as hf:
        hf.create_dataset("imageUrl", data=np.array(img_file_path.encode("utf-8"), dtype='S'))
        hf.create_dataset("pts/responses", data=np.array([p.response for p in pts], dtype=np.float32))
        hf.create_dataset("pts/locations", data=np.array([p.pt for p in pts], dtype=np.float32))
        hf.create_dataset("pts/sizes", data=np.array([p.size for p in pts], dtype=np.float32))
        hf.create_dataset("pts/octaves", data=np.array([p.octave for p in pts], dtype=np.float32))
        hf.create_dataset("des", data=np.array(des, dtype=np.float32))


def compute_and_save_tile_features_split_block(tilespec: dict, output_h5_file_path: str, features_type: str,
                                               features_args: dict, block: int):
    """
    First split the image into several blocks, then compute and save features as H5PY file according to the tilespec
    Commonly used for the singlebeam data with large tile size
    :param tilespec:
    :param output_h5_file_path:
    :param features_type:
    :param features_args:
    :param block:
    :return:
    """
    if features_type not in extract_features.__dict__:
        log_controller.error('Features type not defined. Please refer to common/extract_features.py')
        raise TypeError('features type')
    img_file_path = tilespec["mipmapLevels"]["0"]["imageUrl"]
    img_gray = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
    log_controller.debug("Computing {} features for image: {}".format(features_type, os.path.basename(img_file_path)))

    dims = img_gray.shape
    log_controller.debug("Image dimension: {}. Split in {} blocks".format(dims, block))
    mask_range = int(dims[1] / block)
    mask = np.zeros(dims, dtype=np.uint8)
    pts = []
    des = []
    for i in range(block):
        log_controller.debug("Computing block {}/{}".format(i + 1, block))
        mask[:, (i * mask_range):((i + 1) * mask_range - 1)] = 255
        if i == 0:
            x_start = i * mask_range
        else:
            x_start = i * mask_range - 10
        x_end = (i + 1) * mask_range - 1
        pts_block, des_block = extract_features.__dict__[features_type](img_gray[:, x_start:x_end], features_args)
        for p in pts:
            p_list = list(p.pt)
            p_list[0] = p_list[0] + x_start
            p.pt = tuple(p_list)
        if pts is not None:
            pts.extend(pts_block)
        if des is not None:
            des.extend(des_block)

    if des is None:
        pts = []
        des = []
        log_controller.warning('No feature point detected for {}.'.format(img_file_path))
    log_controller.debug("Saving {} {} features at: {}".format(len(pts), features_type, output_h5_file_path))
    with h5py.File(output_h5_file_path, 'w') as hf:
        hf.create_dataset("imageUrl", data=np.array(img_file_path.encode("utf-8"), dtype='S'))
        hf.create_dataset("pts/responses", data=np.array([p.response for p in pts], dtype=np.float32))
        hf.create_dataset("pts/locations", data=np.array([p.pt for p in pts], dtype=np.float32))
        hf.create_dataset("pts/sizes", data=np.array([p.size for p in pts], dtype=np.float32))
        hf.create_dataset("pts/octaves", data=np.array([p.octave for p in pts], dtype=np.float32))
        hf.create_dataset("des", data=np.array(des, dtype=np.float32))
