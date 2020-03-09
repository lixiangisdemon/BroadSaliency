from supercrf cimport CrfInference
from libcpp.vector cimport vector
import numpy as np

def SuperCRF(img, mask, kernel, seg_img):
    cdef Mat img_mat = np2Mat(img)
    cdef Mat mask_mat = np2Mat(mask)
    cdef Mat kernel_mat = np2Mat(kernel)
    cdef Mat seg = np2Mat(seg_img)
    cdef Mat_[int] seg_mat = Mat2Mat_[int](seg)
    cdef Mat res = CrfInference(img_mat, mask_mat, kernel_mat, seg_mat)
    return Mat2np(res)
