'''Normalize image intensity for classification'''

import enum
import numpy as np
import skimage
from skimage.exposure import equalize_adapthist

class NormalizeMethod(enum.Enum):
    '''The algorithm to use to normalize image planes'''

    '''Use a local adaptive histogram filter to normalize'''
    EQUALIZE_ADAPTHIST=1,
    '''Rescale to -.5, .5, discarding outliers'''
    RESCALE=2,
    '''Rescale 0-255 to 0-1 and otherwise do no normalization'''
    NONE=3,
    '''Match histogram of tile against that of whole image'''
    MATCH=4

def normalize_image_adapthist(img, offset=.5):
    '''Normalize image using a locally adaptive histogram
    
    :param img: image to be normalized
    :returns: normalized image
    '''
    version = tuple(map(int, skimage.__version__.split(".")))
    if version < (0, 12, 0):
        img = img.astype(np.uint16)
    img = equalize_adapthist(img)
    if version < (0, 12, 0):
        # Scale image if prior to 0.12
        imax = img.max()
        imin = img.min()
        img = (img.astype(np.float32) - imin) / \
            (imax - imin + np.finfo(np.float32).eps)
    return img - offset

def normalize_image_rescale(img, saturation_level=0.05, offset=.5):
    '''Normalize the image by rescaling after discaring outliers
    
    :param img: the image to normalize
    :param saturation_level: the fraction of outliers to discard from the
    two extrema
    :param offset: the offset to subtract from the result, scaled to 0-1
    '''
    sortedValues = np.sort( img.ravel())                                        
    minVal = np.float32(
        sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])                                                                      
    maxVal = np.float32(
        sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])                                                                  
    normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))                
    normImg[normImg<0] = 0                                                      
    normImg[normImg>255] = 255                                                  
    return (np.float32(normImg) / 255.0) - offset

#
# The normalization for the histogram matching method was done on the
# Kasthuri AC4 dataset. This will change in the future, but until then,
# these are numbers calculated on AC4
#
'''The fractional cumulative sum of the 256-bin histogram of the AC4 volume'''
uim_quantiles = np.array([
    0.00310061,  0.00440498,  0.00555032,  0.00663368,  0.00767656,
    0.00869805,  0.00970942,  0.01070947,  0.01170988,  0.01271539,
    0.01371903,  0.01472565,  0.01574396,  0.01676395,  0.0177929 ,
    0.01882508,  0.01986628,  0.0209087 ,  0.02196501,  0.02303138,
    0.02410706,  0.02519341,  0.02629363,  0.0274055 ,  0.02853512,
    0.02968522,  0.0308412 ,  0.03200819,  0.03319989,  0.03440769,
    0.03563792,  0.03688377,  0.03816102,  0.03945598,  0.04078296,
    0.04213454,  0.04351143,  0.04491549,  0.0463609 ,  0.04783278,
    0.04934895,  0.05090058,  0.05090058,  0.05248948,  0.054125  ,
    0.0558062 ,  0.05752796,  0.05930098,  0.0611254 ,  0.06299638,
    0.06492598,  0.06691724,  0.06896171,  0.07107004,  0.07324983,
    0.07548998,  0.07780748,  0.08018692,  0.08263973,  0.08516398,
    0.08776816,  0.09044994,  0.09322005,  0.0960726 ,  0.09900747,
    0.10202567,  0.10512624,  0.10830801,  0.11158661,  0.11495885,
    0.11841599,  0.12197106,  0.12560893,  0.12934012,  0.13316071,
    0.13707697,  0.14109526,  0.14518867,  0.14937853,  0.1536528 ,
    0.15802087,  0.16248119,  0.16703333,  0.17166367,  0.17166367,
    0.17638687,  0.18118765,  0.18607435,  0.19105545,  0.19611351,
    0.20125732,  0.20648499,  0.21177473,  0.21714657,  0.22259899,
    0.22812031,  0.23372444,  0.23941671,  0.24516632,  0.25100155,
    0.25690378,  0.26285519,  0.26888803,  0.27499084,  0.28116581,
    0.2874054 ,  0.29371513,  0.30009258,  0.30655565,  0.31306488,
    0.31966661,  0.32630558,  0.33303936,  0.33982559,  0.34667343,
    0.35360199,  0.36058441,  0.3676321 ,  0.37474991,  0.38195053,
    0.38921055,  0.39654446,  0.40393688,  0.41137901,  0.41891357,
    0.42650764,  0.43416054,  0.43416054,  0.44187874,  0.44966515,
    0.45751411,  0.46544174,  0.47341164,  0.48144543,  0.48953384,
    0.49769569,  0.50589523,  0.51414093,  0.5224382 ,  0.53078075,
    0.539189  ,  0.54762028,  0.55610168,  0.56459259,  0.57310806,
    0.5816436 ,  0.5902063 ,  0.59876457,  0.6073621 ,  0.61594475,
    0.62453465,  0.63311203,  0.64168579,  0.65025902,  0.65879616,
    0.66729897,  0.67575653,  0.68416847,  0.69254677,  0.70086403,
    0.70913139,  0.71731743,  0.72544136,  0.73346817,  0.74142937,
    0.74927505,  0.75704323,  0.76470222,  0.77226547,  0.77226547,
    0.77969444,  0.78700714,  0.79421417,  0.80127708,  0.80820435,
    0.81498451,  0.82162949,  0.82811073,  0.83443024,  0.84060791,
    0.84662529,  0.85248856,  0.85819031,  0.86374435,  0.86912201,
    0.87434761,  0.87942329,  0.8843338 ,  0.88909401,  0.89369148,
    0.89815559,  0.90245972,  0.9066111 ,  0.91062599,  0.91448959,
    0.91822899,  0.92181877,  0.92528351,  0.92861458,  0.93181053,
    0.93488434,  0.93782837,  0.94065483,  0.94335876,  0.94595963,
    0.94843666,  0.95080948,  0.95306999,  0.95523628,  0.95729874,
    0.95926689,  0.96115028,  0.96115028,  0.96293602,  0.96465065,
    0.96628387,  0.96785042,  0.96934616,  0.97076797,  0.97213493,
    0.97344238,  0.97469673,  0.97589958,  0.97705765,  0.97817192,
    0.9792392 ,  0.98027466,  0.98127647,  0.98224525,  0.98318115,
    0.98408562,  0.984963  ,  0.98581657,  0.98664909,  0.98746017,
    0.98824966,  0.98901604,  0.98976982,  0.99049309,  0.99120354,
    0.99189186,  0.99256538,  0.9932238 ,  0.993862  ,  0.99447968,
    0.99508301,  0.99566254,  0.99621597,  0.99675003,  0.99725327,
    0.99772911,  0.99816704,  0.9985701 ,  0.99893394,  0.99893394,  1.])
'''the %.1 percentile bin'''
uim_low = 0
'''the 99.9% percentile bin'''
uim_high = 249

def normalize_image_match(img, offset=0):
    '''Match individual planes histograms against that of the global dist'''
    result = []
    for plane in img:
        plane = plane.copy()
        plane[plane < uim_low] = uim_low
        plane[plane > uim_high] = uim_high
        p_bincount = np.bincount(plane.flatten(), minlength=256)
        p_quantiles = \
            np.cumsum(p_bincount).astype(np.float32) / np.prod(plane.shape)
        tbl = np.interp(p_quantiles, uim_quantiles, 
                        np.linspace(-offset, 1-offset, 256).astype(np.float32))
        result.append(tbl[plane])
    
    return np.array(result)

def normalize_image(img, normalize_method, 
                    saturation_level=0.05,
                    offset=.5):
    '''Normalize an image plane's intensity
    
    :param img: the image to normalize
    :param normalize_method: one of the image normalization enumerations
    :param saturation_level: for the rescaling method, the fraction of outliers
    to discard from the distribution (both min and max).
    :param offset: the offset to subtract.
    '''
    if normalize_method == NormalizeMethod.EQUALIZE_ADAPTHIST:
        return np.array([normalize_image_adapthist(_, offset) for _ in img])
    elif normalize_method == NormalizeMethod.RESCALE:
        return np.array([
            normalize_image_rescale(_, saturation_level, offset) for _ in img])
    elif normalize_method == NormalizeMethod.MATCH:
        return normalize_image_match(img)
    else:
        return img.astype(float) / 255.0


