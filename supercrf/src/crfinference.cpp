#include <iostream>
#include "densecrf.h"
#include "crfinference.h"
#include "definition.h"
#include "spGraph.h"
#include "blsCut.h"
using namespace std;

class MatrixFf : public Matrix<float,Dynamic, Dynamic> {
	MatrixFf(int r, int c) : Matrix<float, Dynamic, Dynamic>(r, c) {
	}
	~MatrixFf() {
		cout << "out" << endl;
	}
};
double sigmoid(double x) {
	return 1. / (1 + exp(-x));
}
constexpr float tau = 1.03;
constexpr float epsilon = 1e-8;
void map(const Mat &img, const Mat &anno, Mat &out) {

	int M = 2;
	int cols = img.cols, rows = img.rows;

	MatrixXf u_matrix(M, cols * rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float sal = anno.at<float>(i, j);
			u_matrix(0, i * cols + j) = -log(1 - sal + epsilon) / (tau*sigmoid(1 - sal));
			u_matrix(1, i * cols + j) = -log(sal + epsilon) / (tau*sigmoid(sal));
		}
	}

	int n_iters = 5;
	int sx = 3, sy = 3, sr = 10, sg = 10, sb = 10;
	int X = 60, Y = 60;
	DenseCRF2D crf(img.rows, img.cols, M);
	//ConstUnaryEnergy unary(u_matrix);
	PottsCompatibility potts;
	crf.setUnaryEnergy(u_matrix);
	crf.addPairwiseGaussian(sx, sy, new PottsCompatibility(3));
	crf.addPairwiseBilateral(X, Y, sr, sg, sb, img.data, new PottsCompatibility(10));

	MatrixXf probs = crf.inference(n_iters);

	VectorXs map = crf.map(n_iters);
	//auto map = probs.row(1);
	MatrixXf final_map(rows, cols);
	for (int i = 0, k = 0; i < rows; i++) {
		auto vexs = map.segment(k, cols);
		for (int j = 0; j < cols; j++)
			final_map(i, j) = (float)vexs[j] / (M-1) ;
		k += cols;
	}
	eigen2cv(final_map, out);
}

Mat CrfInference(const Mat& img, const Mat& mask, const Mat& kernel, const Mat_<int>& seg_img) {
	Mat anno = mask.clone();
	Mat img_ = img.clone();
	Mat K = kernel.clone();
	cv::bilateralFilter(img_, img, 3, 40, 50);
	anno.convertTo(anno, CV_32F);
	normalize(anno, anno, 0, 1, NORM_MINMAX);

	img_ = img.clone();
    Mat Lab, HSV;
    cvtColor(img_, Lab, CV_BGR2Lab);
    cvtColor(img_, HSV, CV_BGR2HSV);
    Mat_<Vec3f> color_img, lab_img, hsv_img;
    img_.convertTo(color_img, CV_32FC3, 1.);
    Lab.convertTo(lab_img, CV_32FC3, 1.);
    HSV.convertTo(hsv_img, CV_32FC3, 1.);

	spGraph graph_(seg_img, color_img, lab_img, hsv_img);
	BlsCut cut;
	Mat tmp = cut.blscut(img, anno, seg_img, kernel, graph_);
	Mat out(img.rows, img.cols, CV_32F);
	map(tmp, tmp, out);
	normalize(out, out, 0, 1, cv::NORM_MINMAX);
	out.convertTo(out, CV_8U, 255);
	return out;
}
