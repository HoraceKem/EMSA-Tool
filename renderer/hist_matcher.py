# Given a list of images, creates a "normalized" histogram that can be matched to
# (next step is to take other images and match their histogram to the "normalized" histogram).
import numpy as np
import cv2
import cPickle as pkl

class HistMatcher(object):

    def __init__(self, histogram=None, histogram_fname=None, saturate_low_pct=0.0, saturate_high_pct=0.0):
        # Make sure not both histogram and histogram_fname are given as parameters
        assert(not (histogram is not None and histogram_fname is not None))

        if histogram is None and histogram_fname is None:
            self._histogram = np.zeros((256,), dtype=np.uint32)
        elif histogram is not None:
            self._histogram = histogram
        elif histogram_fname is not None:
            with open(histogram_fname, 'r') as f:
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
        #img_hist, _ = np.histogram(img.flatten(), 256, normed=False)
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        self._histogram += img_hist.astype(np.uint32).T[0]
        self._count += 1
        self._histogram_cdf = None

    def add_image_fname(self, fname):
        img = cv2.imread(fname, 0)
        self.add_image(img)

    def match_histogram(self, img):
        """Match the histogram of the given image to the one stored"""
        if self._r_quantiles is None:
            # Compute the histogram's cdf (based on: http://vzaguskin.github.io/histmatching1/)
            # First normalize the values to [0, 1) s.t. the sum of all values will be 1 (based on "density=True" in the numpy histogram function)
            #self._sat_low_idx = np.argmin(self._histogram_cdf < self._saturate_low_pct)
            #self._sat_high_idx = np.argmax(~(self._histogram_cdf < 1.0 - self._saturate_high_pct))
            # cap from below and above the histogram cdf
            #self._histogram_cdf[:self._sat_low_idx] = self._histogram_cdf[self._sat_low_idx]
            #self._histogram_cdf[self._sat_high_idx + 1:] = self._histogram_cdf[self._sat_high_idx]
            #r_values, r_counts = np.unique(reference, return_counts=True)
            #self._r_values, r_counts = np.unique(self._histogram, return_counts=True)
            self._r_values = np.arange(self._histogram.size)
            #print 'self._r_values', self._r_values
            
            self._r_quantiles = np.cumsum(self._histogram).astype(np.float64) / np.sum(self._histogram)
            #print self._r_quantiles
            #print self._histogram_cdf

        #img_out = np.interp(img.flatten(), range(256), self._histogram_cdf)
        #return img_out.reshape(img.shape)
        # cap the input image by the same values TODO - store them in the "cache"
        #img[img < self._sat_low_idx] = self._sat_low_idx
        #img[img > self._sat_high_idx] = self._sat_high_idx
        img_hist = np.histogram(img.flatten(), 256, [0, 256], density=True)[0]
        img_hist_cdf = img_hist.cumsum()
        interp_r_values = np.interp(img_hist_cdf, self._r_quantiles, self._r_values)
        img_out = interp_r_values[img]#.reshape(orig_shape)
#         orig_shape = img.shape
#         img = img.ravel()
#         s_values, s_idx, s_counts = np.unique(img, return_inverse=True, return_counts=True)
#         print "s_values", len(s_values), s_values
#         s_quantiles = np.cumsum(s_counts).astype(np.float64) / img.size
#         print 's_quantiles', len(s_quantiles), s_quantiles
#         print 'img_hist_cdf', len(img_hist_cdf), img_hist_cdf
#         interp_r_values = np.interp(s_quantiles, self._r_quantiles, self._r_values)
#         interp_r_values2 = np.interp(img_hist_cdf, self._r_quantiles, self._r_values)
#         print interp_r_values
#         print s_idx[:100]
#         img_out = interp_r_values[s_idx].reshape(orig_shape)
#         img_out2 = interp_r_values2[img].reshape(orig_shape)
#         print 'img_out - img_out2:', np.sum(np.abs(img_out - img_out2))
#         print 'img_out[0, :30]', img_out[0, :30]
#         print 'img_out2[0, :30]', img_out2[0, :30]
#         #print img_hist_cdf
        
        return img_out

    def save_histogram(self, fname):
        with open(fname, 'w') as f:
            pkl.dump(self._histogram, f)

   
if __name__ == '__main__':
    #in_ref_dir = '/n/coxfs01/adisuis/image_samples/reference_inverted/'
    in_ref_dir = '/n/coxfs01/adisuis/image_samples/iarpa201610_W01_Sec001/'
    in_fname = '/n/coxfs01/adisuis/image_samples/iarpa201610_W08_Sec115_000011/142_000011_001_2016-09-10T2218095376892.bmp'
    out_fname = '/n/coxfs01/adisuis/image_samples/iarpa201610_W08_Sec115_000011_normalized/142_000011_001_2016-09-10T2218095376892.bmp'
    #out_histogram_fname = '/n/coxfs01/adisuis/image_samples/ac4_inverted_histogram.pkl'
    out_histogram_fname = '/n/coxfs01/adisuis/image_samples/W01_Sec001_histogram.pkl'

    import glob
    import os
    ref_img_fnames = glob.glob(os.path.join(in_ref_dir, '*'))
    norm_hist = HistMatcher(saturate_low_pct=0.001, saturate_high_pct=0.001)
    for fname in ref_img_fnames:
        norm_hist.add_image_fname(fname)

    #equ_hist = np.ones((256,), dtype=np.uint8)
    #norm_hist = HistMatcher(histogram=equ_hist)

    print(norm_hist.histogram)
    in_img = cv2.imread(in_fname, 0)
    out_img = norm_hist.match_histogram(in_img)
    cv2.imwrite(out_fname, out_img)

    norm_hist.save_histogram(out_histogram_fname)

