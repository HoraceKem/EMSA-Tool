import cv2
import numpy as np

__all__ = ['sift', 'surf', 'orb', 'akaze', 'brisk', 'surf_cuda', 'orb_cuda']


def sift(img: np.ndarray, params: dict) -> [tuple, np.ndarray]:
    """
    Extract the SIFT features from an image and return the keypoints and descriptors
    :param img: loaded image in np.ndarray format
    :type img: np.ndarray
    :param params: algorithm parameters
    :type params: dict
    :return: [pts, des]
    """
    sift_created = cv2.SIFT_create(
        params["sift"]["nfeatures"],
        params["sift"]["nOctaveLayers"],
        params["sift"]["contrastThreshold"],
        params["sift"]["edgeThreshold"],
        params["sift"]["sigma"])
    pts, des = sift_created.detectAndCompute(img, None)
    return pts, des


def surf(img: np.ndarray, params: dict) -> [tuple, np.ndarray]:
    """
    Extract the SURF features from an image and return the keypoints and descriptors
    :param img: loaded image in np.ndarray format
    :type img: np.ndarray
    :param params: algorithm parameters
    :type params: dict
    :return: [pts, des]
    """
    surf_created = cv2.xfeatures2d.SURF_create(
        params["surf"]["hessianThreshold"],
        params["surf"]["nOctaves"],
        params["surf"]["nOctaveLayers"],
        params["surf"]["extended"],
        params["surf"]["upright"])
    pts, des = surf_created.detectAndCompute(img, None)
    return pts, des


def orb(img: np.ndarray, params: dict) -> [tuple, np.ndarray]:
    """
    Extract the ORB features from an image and return the keypoints and descriptors
    :param img: loaded image in np.ndarray format
    :type img: np.ndarray
    :param params: algorithm parameters
    :type params: dict
    :return: [pts, des]
    """
    score_type_list = [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE]
    orb_created = cv2.ORB_create(
        params["orb"]["nfeatures"],
        params["orb"]["scaleFactor"],
        params["orb"]["nlevels"],
        params["orb"]["edgeThreshold"],
        params["orb"]["firstLevel"],
        params["orb"]["WTA_K"],
        score_type_list[params["orb"]["scoreType"]],
        params["orb"]["patchSize"],
        params["orb"]["fastThreshold"])
    pts, des = orb_created.detectAndCompute(img, None)
    return pts, des


def akaze(img: np.ndarray, params: dict) -> [tuple, np.ndarray]:
    """
    Extract the AKAZE features from an image and return the keypoints and descriptors
    :param img: loaded image in np.ndarray format
    :type img: np.ndarray
    :param params: algorithm parameters
    :type params: dict
    :return: [pts, des]
    """
    descriptor_type = [cv2.AKAZE_DESCRIPTOR_KAZE, cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT,
                       cv2.AKAZE_DESCRIPTOR_MLDB, cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT]
    diffusivity = [cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_WEICKERT, cv2.KAZE_DIFF_CHARBONNIER]
    akaze_created = cv2.AKAZE_create(
        descriptor_type[params["akaze"]["descriptor_type"]],
        params["akaze"]["descriptor_size"],
        params["akaze"]["descriptor_channels"],
        params["akaze"]["threshold"],
        params["akaze"]["nOctaves"],
        params["akaze"]["nOctaveLayers"],
        diffusivity[params["akaze"]["diffusivity"]],)
    pts, des = akaze_created.detectAndCompute(img, None)
    return pts, des


def brisk(img: np.ndarray, params: dict) -> [tuple, np.ndarray]:
    """
    Extract the BRISK features from an image and return the keypoints and descriptors
    :param img: loaded image in np.ndarray format
    :type img: np.ndarray
    :param params: algorithm parameters
    :type params: dict
    :return: [pts, des]
    """
    brisk_created = cv2.BRISK_create(
        params["brisk"]["thresh"],
        params["brisk"]["octaves"],
        params["brisk"]["patternScale"]
    )
    pts, des = brisk_created.detectAndCompute(img, None)
    return pts, des


def surf_cuda(img: np.ndarray, params: dict) -> [tuple, np.ndarray]:
    """
    Powered by CUDA, and extract the SURF features from an image and return the keypoints and descriptors
    !!! Note that you have to set 'WITH_CUDA = ON' when building OpenCV from source code !!!
    In surf_cuda, there are some APIs different from the style of written of the official C++ version because the APIs
    in python version are not complete. i.e. downloadDescriptor, CUDA_descriptorMatcher
    Please see https://docs.opencv.org/master/index.html for more information about the APIs.
    If you want to see the official style of written, please refer to our C++ version of feature extraction benchmark:
    [https://github.com/HoraceKem/FeaturesBenchmark]
    :param img: loaded image in np.ndarray format
    :type img: np.ndarray
    :param params: algorithm parameters
    :type params: dict
    :return: [pts, des]
    """
    img_gpu = cv2.cuda_GpuMat()
    img_gpu.upload(img)
    surf_cuda_created = cv2.cuda.SURF_CUDA_create(
        params["surf_cuda"]["hessianThreshold"],
        params["surf_cuda"]["nOctaves"],
        params["surf_cuda"]["nOctaveLayers"],
        params["surf_cuda"]["extended"],
        params["surf_cuda"]["keypointsRatio"],
        params["surf_cuda"]["upright"])
    pts_gpu, des_gpu = surf_cuda_created.detectWithDescriptors(img_gpu, None)
    pts = surf_cuda_created.downloadKeypoints(pts_gpu)
    des = des_gpu.download()
    return pts, des


def orb_cuda(img: np.ndarray, params: dict) -> [tuple, np.ndarray]:
    """
    Powered by CUDA, and extract the ORB features from an image and return the keypoints and descriptors
    In orb_cuda, there are some APIs different from the style of written of the official C++ version because the APIs
    in python version are not complete. i.e. detectAndComputeAsync, convert
    Please see https://docs.opencv.org/master/index.html for more information about the APIs.
    If you want to see the official style of written, please refer to our C++ version of feature extraction benchmark:
    [https://github.com/HoraceKem/FeaturesBenchmark]
    :param img: loaded image in np.ndarray format
    :type img: np.ndarray
    :param params: algorithm parameters
    :type params: dict
    :return: [pts, des]
    """
    img_gpu = cv2.cuda_GpuMat()
    img_gpu.upload(img)
    score_type_list = [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE]
    orb_cuda_created = cv2.cuda.ORB_create(
        params["orb_cuda"]["nfeatures"],
        params["orb_cuda"]["scaleFactor"],
        params["orb_cuda"]["nlevels"],
        params["orb_cuda"]["edgeThreshold"],
        params["orb_cuda"]["firstLevel"],
        params["orb_cuda"]["WTA_K"],
        score_type_list[params["orb_cuda"]["scoreType"]],
        params["orb_cuda"]["patchSize"],
        params["orb_cuda"]["fastThreshold"],
        params["orb_cuda"]["blurForDescriptor"])
    pts_gpu, des_gpu = orb_cuda_created.detectAndComputeAsync(img_gpu, None)
    pts = orb_cuda_created.convert(pts_gpu)
    des = des_gpu.download()
    return pts, des
