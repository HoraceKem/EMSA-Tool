import sys
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
#import pylab

FAIL_PMCC_SCORE_TOO_LOW = 0
FAIL_PMCC_ON_EDGE = 1
FAIL_PMCC_CURVATURE_TOO_HIGH = 2
FAIL_PMCC_MAXRATIO_TOO_HIGH = 3
FAIL_PMCC_NOT_LOCALIZED = 4

def PMCC_match(image, template, min_correlation=0.2, maximal_curvature_ratio=10, maximal_ROD=0.9):
    # compute the correlation image
    correlation_image = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # pylab.imshow((correlation_image + 1) / 2.0)

    # find local maxima
    maxima_mask = (correlation_image == maximum_filter(correlation_image, size=3))
    maxima_values = correlation_image[maxima_mask]
    maxima_values.sort()

    if maxima_values[-1] < min_correlation:
        return None, FAIL_PMCC_SCORE_TOO_LOW, 0

    # TrakEM2 code uses (1 + 2nd_best) / (1 + best) for this test...?
    if (maxima_values.size > 1) and (maxima_values[-2] / maxima_values[-1] > maximal_ROD):
        return None, FAIL_PMCC_MAXRATIO_TOO_HIGH, 0

    # find the maximum location
    mi, mj = np.unravel_index(np.argmax(correlation_image), correlation_image.shape)
    if (mi == 0) or (mj == 0) or (mi == correlation_image.shape[0] - 1) or (mj == correlation_image.shape[1] - 1):
        return None, FAIL_PMCC_ON_EDGE, 0

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
    if (det <= 0) or (trace * trace / det > maximal_curvature_ratio):
        return None, FAIL_PMCC_CURVATURE_TOO_HIGH, 0

    # localize by Taylor expansion
    # invert Hessian
    ixx = dyy / det
    ixy = -dxy / det
    iyy = dxx / det

    # calculate offset
    ox = -ixx * dx - ixy * dy
    oy = -ixy * dx - iyy * dy

    if abs(ox) >= 1 or abs(oy) >= 1:
        return None, FAIL_PMCC_NOT_LOCALIZED, 0

    return True, (mi + oy, mj + ox), maxima_values[-1]

if __name__ == '__main__':
    # template = cv2.imread(sys.argv[1], 0)  # flags=0 -> grayscale
    image = cv2.imread(sys.argv[1], 0)
    image_resized = cv2.resize(image, ((image.shape[0] / 2, image.shape[1] / 2)))
    template = image_resized[475:, 315:][:50, :50].copy()

    # force .5 pixel shift
    image = image[1:, 1:]
    image = cv2.resize(image, ((image.shape[0] / 2, image.shape[1] / 2)))

    print(PMCC_match(image, template))
