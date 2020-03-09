import numpy as np
cimport numpy as np  # for np.ndarray
from libc.string cimport memcpy
from opencv_mat cimport *
from libc.stdio cimport printf

# inspired and adapted from http://makerwannabe.blogspot.ch/2013/09/calling-opencv-functions-via-cython.html

cdef Mat np2Mat3D(np.ndarray ary):
    assert ary.ndim==3 and ary.shape[2]==3, "ASSERT::3channel RGB only!!"
    ary = np.dstack((ary[...,0], ary[...,1], ary[...,2])) #RGB -> BGR

    cdef np.ndarray[np.uint8_t, ndim=3, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.uint8)
    cdef uchar* im_buff = <uchar*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_8UC3)
    memcpy(m.data, im_buff, r*c*3)
    return m

cdef Mat np2Mat2D(np.ndarray ary):
    assert ary.ndim==2 , "ASSERT::1 channel grayscale only!!"

    cdef np.ndarray[np.uint8_t, ndim=2, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.uint8)
    cdef uchar* im_buff = <uchar*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_8U)
    memcpy(m.data, im_buff, r*c)
    return m

cdef Mat np2Mat2D_F32(np.ndarray ary):
    assert ary.ndim==2 , "ASSERT::1 channel grayscale only!!"
    assert ary.dtype==np.float32, "ASSERT dtype=float32"

    cdef np.ndarray[np.float32_t, ndim=2, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.float32)
    cdef float* im_buff = <float*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_32F)
    memcpy(m.data, im_buff, r*c*sizeof(float)) # 4 is the size of 
    return m

def npto32ftonp(nparr):
    assert nparr.dtype == np.float32, "array dtype must be float32"
    return Mat2np(np2Mat2D_F32(nparr))

cdef Mat np2Mat2D_S32(np.ndarray ary):
    assert ary.ndim==2 , "ASSERT::1 channel grayscale only!!"
    assert ary.dtype==np.int32, "ASSERT dtype=int32"

    cdef np.ndarray[np.int32_t, ndim=2, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.int32)
    cdef int* im_buff = <int*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef Mat m
    m.create(r, c, CV_32S)
    memcpy(m.data, im_buff, r*c*sizeof(int)) # 4 is the size of 
    return m

cdef Mat np2Mat(np.ndarray ary):
    cdef Mat out
    if ary.ndim == 2:
        if ary.dtype == np.float32:
            out = np2Mat2D_F32(ary)
        elif ary.dtype == np.uint8:
            out = np2Mat2D(ary)
        elif ary.dtype == np.int32:
            out = np2Mat2D_S32(ary)
        else:
            raise TypeError("array data type is not valid")
    elif ary.ndim == 3:
        out = np2Mat3D(ary)
    return out

cdef object Mat2np(Mat m):

    shape = (m.rows, m.cols, m.channels())
    cdef size_t sz = shape[0] * shape[1] * shape[2]
    cdef float[:,:,::1] data1
    cdef int[:,:,::1] data2
    cdef uchar[:,:,::1] data3
    
    if m.depth() == CV_32F:
        ary = np.zeros(shape=shape, dtype=np.float32)
        sz *= sizeof(float)
        data1 = ary
        memcpy(&data1[0,0,0], m.data, sz)
        if m.channels() == 1:
            return np.asarray(data1).squeeze(axis=-1)
        else:
            return np.asarray(data1)
    elif m.depth() == CV_32S:
        ary = np.zeros(shape=shape, dtype=np.int32)
        sz *= sizeof(int)
        data2 = ary
        memcpy(&data2[0,0,0], m.data, sz)
        if m.channels() == 1:
            return np.asarray(data2).squeeze(axis=-1)
        else:
            return np.asarray(data2)
    elif m.depth() == CV_8U:
        ary = np.zeros(shape=shape, dtype=np.uint8)
        data3 = ary
        memcpy(&data3[0,0,0], m.data, sz)
        if m.channels() == 1:
            return np.asarray(data3).squeeze(axis=-1)
        else:
            return np.asarray(data3)


def np2Mat2np(nparray):
    cdef Mat m

    # Convert numpy array to cv::Mat
    m = np2Mat(nparray)

    # Convert cv::Mat to numpy array
    pyarr = Mat2np(m)

    return pyarr


cdef class PyMat:
    def __cinit__(self, np_mat):
        self.mat = np2Mat(np_mat)

    def get_mat(self):
        return Mat2np(self.mat)


cdef class PyMat8U:
    def __cinit__(self, np_mat):
        cdef Mat mat = np2Mat(np_mat)
        self.mat_u = Mat2Mat_[uchar](mat)
        self.mat_o = Mat_2Mat(self.mat_u)

    @property
    def rows(self):
        return self.mat_o.rows
    @property
    def cols(self):
        return self.mat_o.cols

    def channels(self):
        return self.mat_o.channels()

    def depth(self):
        return self.mat_o.depth()
    
    def elemSize(self):
        return self.mat_o.elemSize()

    def printValues(self):
        for i in xrange(self.rows):
            for j in xrange(self.cols):
                printf("%d ", self.mat_o.data[i * self.cols + j])
            printf("\n")
    def printNumpy(self):
        array = Mat2np(self.mat_o)
        print array

    