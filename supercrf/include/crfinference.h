#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include "densecrf.h"

using namespace std;
using cv::Mat;
using cv::Mat_;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Dense;
using Eigen::Dynamic;
using Eigen::MatrixXf;

template<class T>
inline void cv2eigen(const Mat &src, Matrix<T, Dynamic, Dynamic> &dst) {
	CV_Assert(src.data != NULL && dst.data() != NULL);
	CV_Assert(src.rows == dst.rows() && src.cols == dst.cols());
}

template<>
inline void cv2eigen(const Mat &src, MatrixXf &dst) {
	CV_Assert(src.data != NULL && dst.data() != NULL);
	CV_Assert(src.rows == dst.rows() && src.cols == dst.cols());
	for (int i = 0; i < src.rows; i++) {
		const uchar *data = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			dst(i, j) = (float)data[j];
		}
	}
}

template<class T>
inline void eigen2cv(const Matrix<T, Dynamic, Dynamic> &src, Mat &dst) {
	CV_Assert(src.data() != NULL && dst.data != NULL);
	CV_Assert(src.rows() == dst.rows && src.cols() == dst.cols);
}

template<>
inline void eigen2cv(const MatrixXf &src, Mat &dst) {
	CV_Assert(src.data() != NULL && dst.data != NULL);
	CV_Assert(src.rows() == dst.rows && src.cols() == dst.cols);
	for (int i = 0; i < src.rows(); i++) {
		float *data = dst.ptr<float>(i);
		for (int j = 0; j < src.cols(); j++) {
			data[j] = src(i, j);
		}
	}
}

Mat CrfInference(const Mat& img, const Mat& mask, const Mat& kernel, const Mat_<int>& seg_img);
