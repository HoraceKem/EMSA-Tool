import sys
#import os
import cv2
import numpy as np
#import ujson as json
import pickle
import skimage.exposure


class HistogramCLAHE(object):

    def __init__(self):
        pass

    def adjust_histogram(self, img_path, img):
        img1_equ = ((skimage.exposure.equalize_adapthist(img)) * 255).astype(np.uint8)
        return img1_equ


class HistogramGB11CLAHE(object):

    def __init__(self):
        pass

    def adjust_histogram(self, img_path, img):
        img = cv2.GaussianBlur(img, (11,11), 0)
        img1_equ = ((skimage.exposure.equalize_adapthist(img)) * 255).astype(np.uint8)
        return img1_equ


if __name__ == '__main__':
    
    #in_json_fname = 'W08_Sec214_montaged_2tiles.json'
    #out_pkl_fname = 'W08_Sec214_montaged_2tiles_adjuster.pkl'
    in_json_fname = sys.argv[1]
    out_pkl_fname = sys.argv[2]

    clahe = HistogramCLAHE()
    with open(out_pkl_fname, 'wb') as out_f:
        pickle.dump(clahe, out_f, pickle.HIGHEST_PROTOCOL)

    
