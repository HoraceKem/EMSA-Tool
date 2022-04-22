import cv2
import numpy as np

__all__ = ['bf', 'flann']


def bf(des1: np.ndarray, des2: np.ndarray, params: dict) -> list:
    """
    Brute force match and return the matches.
    The distance type is decided by the type of descriptors.
    e.g. SIFT features' descriptor is float32, so we use L2 distance
    :param des1:
    :type des1: np.ndarray
    :param des2:
    :param params:
    :type des2: np.ndarray
    :return: a list of matches
    """
    if des1.shape[0] <= 2 or des2.shape[0] <= 2:
        return ['NO_MATCH']
    if des1.dtype == np.float32:
        dist = cv2.NORM_L2
    else:
        dist = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(dist)
    all_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []

    for m, n in all_matches:
        if m.distance < params["bf"]["threshold"] * n.distance:
            good_matches.append([m])
    if len(good_matches) <= 2:
        return ['NO_MATCH']
    else:
        return good_matches


def flann(des1, des2, params):
    """
    FLANN-based match and return the matches.
    :param des1:
    :type des1: np.ndarray
    :param des2:
    :param params:
    :type des2: np.ndarray
    :return: a list of matches
    """
    if des1.shape[0] <= 2 or des2.shape[0] <= 2:
        return ['NO_MATCH']
    # FLANN-based matching only works with np.float32 type
    if not des1.dtype == np.float32:
        des1 = np.array(des1, dtype=np.float32)
        des2 = np.array(des2, dtype=np.float32)
    matcher = cv2.FlannBasedMatcher()
    all_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in all_matches:
        if m.distance < params["flann"]["threshold"] * n.distance:
            good_matches.append([m])
    if len(good_matches) <= 2:
        return ['NO_MATCH']
    else:
        return good_matches
