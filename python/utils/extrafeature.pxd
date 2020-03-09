from libcpp.vector cimport vector
from libcpp.pair cimport pair
from opencv_mat cimport *

cdef extern from "spStat.h":
    cdef cppclass c_spStat "spStat":
        c_spStat() except +
        void extractFeatures(const Mat & img, const Mat & gt_, Mat_[int]& seg_img, Mat_[float]& feats, vector[float]& labels)
        void distFeatures(const Mat & img, const Mat & gt_, Mat_[int]& lbls, Mat_[float] &feats1, Mat_[float] &feats2, 
            vector[float] &dists)
        void clear()
        pair[Mat_[float], vector[float]] interpolating(const Mat_[float]& features, const vector[float]& labels, int size)
        void backproject(const Mat_[float]& features, const vector[float]& nlabels, vector[float]& labels, int size)

        @staticmethod
        void gen_map(Mat &res, const vector[float]& labels, const Mat_[int] &seg_img)

cdef class ExtraFeature:
    cdef c_spStat* c_stat 
    cdef int feat_size
