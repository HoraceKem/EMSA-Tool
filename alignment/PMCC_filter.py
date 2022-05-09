import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter


def PMCC_match(image, template, align_args):
    """
    PMCC match
    :param image:
    :param template:
    :param align_args:
    :return:
    """
    # compute the correlation image
    correlation_image = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # find local maxima
    maxima_mask = (correlation_image == maximum_filter(correlation_image, size=3))
    maxima_values = correlation_image[maxima_mask]
    maxima_values.sort()

    if maxima_values[-1] < align_args["PMCC"]["min_correlation"]:
        return None, align_args["PMCC"]["fail_PMCC_score_too_low"], 0

    # TrakEM2 code uses (1 + 2nd_best) / (1 + best) for this test...?
    if (maxima_values.size > 1) and (maxima_values[-2] / maxima_values[-1] > align_args["PMCC"]["maximal_ROD"]):
        return None, align_args["PMCC"]["fail_PMCC_max_ratio_too_high"], 0

    # find the maximum location
    mi, mj = np.unravel_index(np.argmax(correlation_image), correlation_image.shape)
    if (mi == 0) or (mj == 0) or (mi == correlation_image.shape[0] - 1) or (mj == correlation_image.shape[1] - 1):
        return None, align_args["PMCC"]["fail_PMCC_on_edge"], 0

    # extract pixels around maximum
    [[c00, c01, c02],
     [c10, c11, c12],
     [c20, c21, c22]] = correlation_image[(mi - 1):(mi + 2),
                                          (mj - 1):(mj + 2)]

    dx = (c12 - c10) / 2.0
    dy = (c21 - c01) / 2.0
    dxx = c10 - c11 - c11 + c12
    dyy = c01 - c11 - c11 + c21
    dxy = (c22 - c20 - c02 + c00) / 4.0

    det = dxx * dyy - dxy * dxy
    trace = dxx + dyy
    if (det <= 0) or (trace * trace / det > align_args["PMCC"]["maximal_curvature_ratio"]):
        return None, align_args["PMCC"]["fail_PMCC_curvature_too_high"], 0

    # localize by Taylor expansion
    # invert Hessian
    ixx = dyy / det
    ixy = -dxy / det
    iyy = dxx / det

    # calculate offset
    ox = -ixx * dx - ixy * dy
    oy = -ixy * dx - iyy * dy

    if abs(ox) >= 1 or abs(oy) >= 1:
        return None, align_args["PMCC"]["fail_PMCC_not_localized"], 0

    return True, (mi + oy, mj + ox), maxima_values[-1]
