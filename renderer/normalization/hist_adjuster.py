# The interface for histogram adjustment has a single method:
#   adjust_histogram(img_path, img) - that receives an image path (or a unique image key) and an image as input and reutrns an adjusted image as output
#
# The adjusters are objects that have the adjust_histogram method implemented
# 

import cPickle as pickle

def load_adjuster(pkl_fname):
    with open(pkl_fname, 'rb') as in_f:
        adjuster = pickle.load(in_f)

    return adjuster


