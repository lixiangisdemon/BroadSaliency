from libcpp.vector cimport vector
from opencv_mat cimport *

cdef extern from "crfinference.h":
    Mat CrfInference(const Mat& img, const Mat& mask, const Mat& kernel, const Mat_[int]& seg_img)