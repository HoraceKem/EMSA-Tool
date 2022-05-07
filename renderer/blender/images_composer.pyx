#wraps ImagesComposer

# largely based on:
#   https://bitbucket.org/petters/hakanardo/src/0f994ec688cfba42a4e019fa80c189a13f28b109/copencv/libopencv.pxd?at=default&fileviewer=file-view-default
#   https://bitbucket.org/petters/hakanardo/src/0f994ec688cfba42a4e019fa80c189a13f28b109/copencv/copencv.pyx?at=default&fileviewer=file-view-default

import numpy as np
cimport numpy as np
cimport cython
import cv2
cimport numpy as np # for np.ndarray
from libcpp.string cimport string
from cython.operator import dereference
#from libc.string cimport memcpy
from libcpp.vector cimport vector
from libcpp cimport bool

#ctypedef void* int_parameter
#ctypedef int_parameter two "2"
#ctypedef Point_[float, two] Point2f


cdef class WrappedMat:
    cdef Mat* _mat
    def __cinit__(self):
        self._mat = NULL
    def __dealloc__(self): 
        if self._mat:            
            del self._mat
            self._mat = NULL

class CBufferView(np.ndarray):
    pass

np.import_array()

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat()
        Mat(Mat& m)
        Mat(int _rows, int _cols, int _type)
        Mat(int rows, int cols, int type, void* data)        
        int dims, rows, cols
        unsigned char *data
        bool isContinuous()
        int type()
        int channels()

    cdef cppclass _InputArray:
        _InputArray()
        _InputArray(Mat& m)
        _InputArray(vector[Mat]& vec)
        
    ctypedef _InputArray& InputArray

    ctypedef InputArray InputArrayOfArrays

    cdef cppclass _OutputArray(_InputArray):
        _OutputArray()
        _OutputArray(Mat& m)
        Mat getMat(int idx)
        
    ctypedef _OutputArray& OutputArray

    cdef cppclass Size_[_Tp]:
        Size_()
        Size_(_Tp _width, _Tp _height)
        Size_(Size_& sz)
    #ctypedef Size_[int] Size2i
    #ctypedef Size2i Size

    cdef cppclass Point_[_Tp]:
        _Tp x, y
        Point_(_Tp _x, _Tp _y)

    #ctypedef Point_[float] Point2f
    ctypedef Point_[int] Point

cdef extern from "opencv2/core/types_c.h":
    enum:
        CV_8UC1
        CV_8UC2
        CV_8UC3
        CV_8UC4
        CV_32FC1
        CV_64FC1
        CV_32SC1
        CV_32SC2
        CV_32SC3
        CV_32SC4

# cdef extern from "opencv2/features2d/features2d.hpp" namespace "cv":
#     cdef cppclass KeyPoint:
#         Point_[float] pt
#         float size
#         float angle


# cdef extern from "opencv2/core/types.hpp" namespace "cv":
#     struct KeyPoint:
#         pass
# 
# cdef extern from "opencv2/core/mat.hpp" namespace "cv":
#     cdef cppclass Mat:
#         Mat() except +
#         void create(int, int, int)
#         void* data
#     #struct InputArray:
#     #    pass

# cdef void ary2cvMat(np.ndarray ary, Mat& out):
#     assert(ary.ndim==2 and ary.shape[2]==2, "ASSERT::2channel grayscale only!!")
#      
#     cdef np.ndarray[np.uint8_t, ndim=2, mode = 'c'] np_buff = np.ascontiguousarray(ary, dtype = np.uint8)
#     cdef unsigned int* im_buff = <unsigned int*> np_buff.data
#     cdef int r = ary.shape[0]
#     cdef int c = ary.shape[1]
#     out.create(r, c, CV_8UC3)
#     memcpy(out.data, im_buff, r*c*3)

cdef mat2numpy(Mat mat):
    cdef np.npy_intp dims[3]
    assert mat.isContinuous()
    assert mat.dims == 2
    if mat.type() == CV_8UC3:
        dims[0], dims[1], dims[2] = mat.rows, mat.cols, 3
        ar = np.PyArray_SimpleNewFromData(3, dims, np.NPY_UINT8, mat.data)
    elif mat.type() == CV_64FC1:
        dims[0], dims[1] = mat.rows, mat.cols
        ar = np.PyArray_SimpleNewFromData(2, dims, np.NPY_FLOAT64, mat.data)
    elif mat.type() == CV_32SC1:
        dims[0], dims[1] = mat.rows, mat.cols
        ar = np.PyArray_SimpleNewFromData(2, dims, np.NPY_INT32, mat.data)
    elif mat.type() == CV_8UC1:
        dims[0], dims[1] = mat.rows, mat.cols
        ar = np.PyArray_SimpleNewFromData(2, dims, np.NPY_UINT8, mat.data)
    else:
        assert False, "Unknown type %d" %  mat.type()
    ar = ar.view(CBufferView)
    wmat = WrappedMat()
    wmat._mat = new Mat(mat)
    ar._wmat = wmat
    return ar


cdef Mat numpy2mat(img):
    cdef WrappedMat wmat
    if hasattr(img, '_wmat'):
        wmat = img._wmat
        return dereference(wmat._mat)
    else:
        assert img.flags.contiguous
        if len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == 'B':
            #print("3channels image - CV_8UC3", img.dtype)
            return Mat(img.shape[0], img.shape[1], CV_8UC3, np.PyArray_DATA(img))
        elif len(img.shape) == 2 and img.dtype == 'd':
            #print("2channels image - CV64_FC1", img.dtype)
            return Mat(img.shape[0], img.shape[1], CV_64FC1, np.PyArray_DATA(img))
        elif len(img.shape) == 2 and img.dtype == 'B':
            #print("2channels image - CV_8UC1", img.dtype)
            return Mat(img.shape[0], img.shape[1], CV_8UC1, np.PyArray_DATA(img))
        else:
            print("Unknown image properties: shape:", img.shape, "type:", img.dtype)
            assert False

# cdef populate_img_vector(in_list, out_vec):
#     for element in in_list:
#         out_vec.push_back(InputArray(numpy2mat(element)))
# 
# cdef populate_point_vector(in_list, out_vec):
#     for element in in_list:
#         out_vec.push_back(Point(element[0], element[1]))


cdef extern from "ImagesComposer.hpp":

    cdef cppclass ImagesComposer:
        @staticmethod
        int compose_panorama(
            InputArrayOfArrays in_warped_images, InputArrayOfArrays in_warped_masks, vector[Point]& in_warped_corners,
            float seam_scale,
            InputArrayOfArrays in_warped_seams_images, InputArrayOfArrays in_warped_seams_masks, vector[Point]& in_warped_seams_corners,
            OutputArray pano)

cdef class PyImagesComposer:

    @staticmethod
    def compose_panorama(warped_images, warped_masks, warped_corners,
                         seam_scale, warped_seams_images, warped_seams_masks, warped_seams_corners):
        # Create a matrix for the output image
        cdef Mat out
        cdef vector[Mat] in_warped_images
        cdef vector[Mat] in_warped_masks
        cdef vector[Point] in_warped_corners
        cdef vector[Mat] in_warped_seams_images
        cdef vector[Mat] in_warped_seams_masks
        cdef vector[Point] in_warped_seams_corners

        #populate_img_vector(warped_images, in_warped_images)
        #print("populate1")
        for element in warped_images:
            in_warped_images.push_back(numpy2mat(element))
        #print("populate2")
        for element in warped_masks:
            in_warped_masks.push_back(numpy2mat(element))
        #print("populate3")
        for element in warped_corners:
            in_warped_corners.push_back(Point(element[0], element[1]))

        #print("populate4")
        for element in warped_seams_images:
            in_warped_seams_images.push_back(numpy2mat(element))
        #print("populate5")
        for element in warped_seams_masks:
            in_warped_seams_masks.push_back(numpy2mat(element))
        #print("populate6")
        for element in warped_seams_corners:
            in_warped_seams_corners.push_back(Point(element[0], element[1]))

#         populate_img_vector(warped_masks, in_warped_masks)
#         populate_point_vector(warped_corners, in_warped_corners)
#         populate_img_vector(warped_seams_images, in_warped_seams_images)
#         populate_img_vector(warped_seams_masks, in_warped_seams_masks)
#         populate_point_vector(warped_seams_corners, in_warped_seams_corners)

        #print("compose panorma called")
        ImagesComposer.compose_panorama(
            InputArray(in_warped_images), InputArray(in_warped_masks), in_warped_corners,
            seam_scale,
            InputArray(in_warped_seams_images), InputArray(in_warped_seams_masks), in_warped_seams_corners,
            OutputArray(out))
        # Return the output converted to a numpy array
        return mat2numpy(out)
        

 
