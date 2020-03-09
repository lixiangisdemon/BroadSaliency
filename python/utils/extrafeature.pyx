from libcpp.vector cimport vector
from libcpp.pair cimport pair
from extrafeature cimport c_spStat
import numpy as np
from libc.stdio cimport printf

cdef class ExtraFeature:

    def __cinit__(self, int feat_size = 40):
        self.c_stat = new c_spStat()
        self.feat_size = feat_size / 2

    def __dealloc__(self):
        if self.c_stat:
            del self.c_stat

    def extract_feature(self, img, Gt = None):
        img = np.array(img, dtype=np.uint8)
        cdef Mat mat, gt
        mat = np2Mat(img)
        if (Gt is None):
            gt.create(mat.rows, mat.cols, CV_8UC3)
        else:
            gt = np2Mat(Gt)
        cdef Mat_[float] features
        cdef Mat_[int] seg_img
        cdef vector[float] labels
        self.c_stat.extractFeatures(mat, gt, seg_img, features, labels)
        return [ Mat2np(Mat_2Mat(seg_img)), Mat2np(Mat_2Mat(features)), labels ]
    
    def cvt_gridfeature(self, feat, labels):
        cdef Mat_[float] features = Mat2Mat_[float](np2Mat(feat))
        cdef pair[Mat_[float], vector[float]] res = self.c_stat.interpolating(features, labels, self.feat_size)
        return [ Mat2np(Mat_2Mat(res.first)), res.second ]
    
    def extract_dists(self, img, Gt = None):
        img = np.array(img, dtype=np.uint8)
        cdef Mat mat, gt
        mat = np2Mat(img)
        if (Gt is None):
            gt.create(mat.rows, mat.cols, CV_8UC3)
        else:
            gt = np2Mat(Gt)

        cdef Mat_[float] feat1, feat2
        cdef Mat_[int] lbl_pair
        cdef vector[float] dist
        self.c_stat.distFeatures(mat, gt, lbl_pair, feat1, feat2, dist)
        self.c_stat.clear()
        return [ Mat2np(Mat_2Mat(feat1)), Mat2np(Mat_2Mat(feat2)), Mat2np(Mat_2Mat(lbl_pair)), dist ]
    
    def getMap(self, seg_img, lbl):
        cdef Mat map
        if (seg_img is None or lbl is None):
            return np.array()
            
        cdef Mat_[int] seg_mat = Mat2Mat_[int](np2Mat(seg_img))
        self.c_stat.gen_map(map, lbl, seg_mat)
        
        return Mat2np(map)
