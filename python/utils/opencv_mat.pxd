cimport numpy as np
import numpy as np

# For cv::Mat usage
cdef extern from "opencv2/core.hpp":
  cdef int  CV_WINDOW_AUTOSIZE
  cdef int CV_8UC3
  cdef int CV_32SC3
  cdef int CV_32FC3
  cdef int CV_8UC1
  cdef int CV_32FC1
  cdef int CV_8U
  cdef int CV_32F
  cdef int CV_32S
  ctypedef unsigned char uchar

cdef extern from "opencv2/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    Mat(int, int, int) except +
    Mat(const Mat &) except +
    void create(int, int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()

  cdef cppclass Mat_[T]:
    Mat_() except +
    Mat_(int, int) except +
    Mat_(const Mat& mat) except +
    Mat_(const Mat_& mat) except +
    void create(int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()

# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO

cdef Mat np2Mat(np.ndarray ary)

cdef object Mat2np(Mat mat)

cdef extern from "opencv_impl.cpp":
  Mat Mat_2Mat[T](Mat_[T] mat)
  Mat_[T] Mat2Mat_[T](Mat mat)


cdef class PyMat:
    cdef Mat mat

cdef class PyMat8U:
    cdef Mat_[uchar] mat_u
    cdef Mat mat_o