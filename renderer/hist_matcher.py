# Given a list of images, creates a "normalized" histogram that can be matched to
# (next step is to take other images and match their histogram to the "normalized" histogram).
import numpy as np
import cv2
import pickle as pkl


class HistMatcher(object):

    def __init__(self, histogram=None, histogram_file_path=None, saturate_low_pct=0.0, saturate_high_pct=0.0):
        # Make sure not both histogram and histogram_file_path are given as parameters
        self._r_values = None
        self._histogram_cdf = None
        assert(not (histogram is not None and histogram_file_path is not None))

        if histogram is None and histogram_file_path is None:
            self._histogram = np.zeros((256,), dtype=np.uint32)
        elif histogram is not None:
            self._histogram = histogram
        elif histogram_file_path is not None:
            with open(histogram_file_path, 'r') as f:
                self._histogram = pkl.load(f)
        self._count = 0
        self._r_quantiles = None
        self._saturate_low_pct = saturate_low_pct
        self._saturate_high_pct = saturate_high_pct

    @property
    def histogram(self):
        return self._histogram

    @property
    def averaged_histogram(self):
        return self._histogram / self._count

    def add_image(self, img):
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        self._histogram += img_hist.astype(np.uint32).T[0]
        self._count += 1
        self._histogram_cdf = None

    def add_image_file_path(self, file_path):
        img = cv2.imread(file_path, 0)
        self.add_image(img)

    def match_histogram(self, img):
        """Match the histogram of the given image to the one stored"""
        if self._r_quantiles is None:
            """
            Compute the histogram's cdf (based on: http://vzaguskin.github.io/histmatching1/)
            First normalize the values to [0, 1) s.t. the sum of all values will be 1 
            (based on "density=True" in the numpy histogram function)
            self._sat_low_idx = np.argmin(self._histogram_cdf < self._saturate_low_pct)
            self._sat_high_idx = np.argmax(~(self._histogram_cdf < 1.0 - self._saturate_high_pct))
            cap from below and above the histogram cdf
            """
            self._r_values = np.arange(self._histogram.size)
            self._r_quantiles = np.cumsum(self._histogram).astype(np.float64) / np.sum(self._histogram)

        img_hist = np.histogram(img.flatten(), 256, [0, 256], density=True)[0]
        img_hist_cdf = img_hist.cumsum()
        interp_r_values = np.interp(img_hist_cdf, self._r_quantiles, self._r_values)
        img_out = interp_r_values[img]

        return img_out

    def save_histogram(self, file_path):
        with open(file_path, 'w') as f:
            pkl.dump(self._histogram, f)


