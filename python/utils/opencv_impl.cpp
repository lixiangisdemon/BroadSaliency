#include <opencv2/core.hpp>
#include <string.h>
#include <Python.h>
using namespace std;
using namespace cv;

template<typename T> 
static Mat Mat_2Mat(Mat_<T> mat);

template<typename T> 
Mat Mat_2Mat(Mat_<T> mat) {
    return Mat();
}

template <> 
Mat Mat_2Mat(Mat_<float> mat) {
    Mat m(mat.rows, mat.cols, CV_32F);
    memcpy(m.data, mat.data, mat.rows * mat.cols * sizeof(float));
    return m;
}

template <> 
Mat Mat_2Mat(Mat_<int> mat) {
    Mat m(mat.rows, mat.cols, CV_32S);
    memcpy(m.data, mat.data, mat.rows * mat.cols * sizeof(int));
    return m;
}

template <> 
Mat Mat_2Mat(Mat_<uchar> mat) {
    Mat m(mat.rows, mat.cols, CV_8U);
    memcpy(m.data, mat.data, mat.rows * mat.cols * sizeof(uchar));
    return m;
}

template<typename T> 
static Mat_<T> Mat2Mat_(Mat mat) {
    Mat_<T> m(mat);
    return m;
}