#include <limits>
#include <cstdlib>
#include <cmath>
#include "grabcut.h"
#include "graph.hpp"
#include <cstring>

/*
 * Gmm - Gaussian Mixture Model
 */

class Gmm {
public:
	static const int componentCount = 5;
	Gmm(Mat & _model);
	double operator() (const Vec3d color) const;
	double operator() (int ci, const Vec3d color) const;
	int whichComponent(const Vec3d color) const;
	
	void initLearning();
	void addSample(int ci, const Vec3d color);
	void endLearning();
private:
	void calcInverseCovAndDeterm(int ci);
	Mat  model;
	double * coefs;
	double * means;
	double * covs;
	
	double inverseCovs[componentCount][3][3];
	double covDeterms[componentCount];

	double sums[componentCount][3];
	double prods[componentCount][3][3];
	int sampleCounts[componentCount];
	int totalSampleCount;
};

Gmm::Gmm(Mat & _model) {
	/* one component is one model whose size is modelSize */
	const int modelSize = 3 /*mean*/ + 9 /*covariance*/ + 1/*component weight*/;
	if (_model.empty()) {
		// there are componentCount models, so the total parameters is componentCount * modelSize
		_model.create(1, modelSize * componentCount, CV_64FC1);
		_model.setTo(Scalar(0));	
	}
	else if ((_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentCount)) {
		std::cerr << "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentCount" << std::endl;
		return ;
	}

	model = _model;
	// Noting the arrangement of parameters. The order is 
	// first componentCount's coefs, then 3 * componentCount's mean 
	// and 3*3*componentCount's cov finaly.
	coefs = model.ptr<double>(0);
	means  = coefs + componentCount; //model.ptr<double>(componentCount);
	covs   = means  + 3*componentCount; //model.ptr<double>(4*componentCount);

	for (int ci = 0; ci < componentCount; ci++) {
		if (coefs[ci] > 0)
			calcInverseCovAndDeterm(ci);
	}

}

/* calculate the probability of one pixel is belong to the Gmm models */
double Gmm::operator()(const Vec3d color) const {
	double res = 0;
	for (int ci = 0; ci < componentCount; ci++) 
		res += coefs[ci] * (*this)(ci, color);
	return res;
}

/* calculate the probability of one pixel is belong to the ci models */
double Gmm::operator()(int ci, const Vec3d color) const {
	double res = 0;
	if (coefs[ci] > 0) {
		assert(covDeterms[ci] > std::numeric_limits<double>::epsilon());
		Vec3d diff = color;
		double *m = means + 3*ci;
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
		double mult = diff[0] * (diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
					+ diff[1] * (diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
					+ diff[2] * (diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
		res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f*mult);
	}
	return res;
}

// return a Gmm index that the pixel is most likely belong to.
int Gmm::whichComponent(const Vec3d color) const {
	int k = 0;
	double max = 0;
	
	for (int ci = 0; ci < componentCount; ci++) {
		double p = (*this)(ci, color);
		if (max < p) {
			k = ci;
			max = p;
		}
	}	
	return k;
}

void Gmm::initLearning() {
	memset(sums, 0.0, componentCount * 3 * sizeof(double));
	memset(prods, 0.0, componentCount * 9 * sizeof(double));
	memset(sampleCounts, 0, componentCount * sizeof(int));
	totalSampleCount = 0;
}

void Gmm::addSample(int ci, const Vec3d color) {
	sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
	prods[ci][0][0]+=color[0]*color[0]; prods[ci][0][1]+=color[0]*color[1]; prods[ci][0][2]+=color[0]*color[2];
	prods[ci][1][0]+=color[1]*color[0]; prods[ci][1][1]+=color[1]*color[1]; prods[ci][1][2]+=color[1]*color[2];
	prods[ci][2][0]+=color[2]*color[0]; prods[ci][2][1]+=color[2]*color[1]; prods[ci][2][2]+=color[2]*color[2];
	sampleCounts[ci]++;
	totalSampleCount++;
}

// Learing parameters from samples, for each gmm model, it learns coefs, means as well as covs
void Gmm::endLearning() {
	const double variance = 0.01; // white noise
	for (int ci = 0; ci < componentCount; ci++) {
		int n = sampleCounts[ci];
		if (n == 0)
			coefs[ci] = 0;
		else {
			coefs[ci] = (double)n / totalSampleCount;
			double *m = means + 3*ci;
			m[0] = sums[ci][0]/n; m[1] = sums[ci][1] / n; m[2] = sums[ci][2] / n;
			double *c = covs + 9*ci;
			c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
			c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
			c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

			double determ = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
			while (determ <= std::numeric_limits<double>::epsilon()) {
				//Adds the white noise to avoid singular covarience matrix
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
				determ = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
			}
			calcInverseCovAndDeterm(ci);
		}
	}
}

/* calculate Inverse covariance and Determinant */
void Gmm::calcInverseCovAndDeterm(int ci) {
	if (coefs[ci] > 0) {
		double *c = covs + 9 * ci;
		double dtrm = 
			covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

		assert(dtrm > std::numeric_limits<double>::epsilon());

		// solve Inverse of covariance
		inverseCovs[ci][0][0] = (c[4]*c[8] - c[5]*c[7]) / dtrm;
		inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
		inverseCovs[ci][2][0] = (c[3]*c[7] - c[4]*c[6]) / dtrm;
		inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
		inverseCovs[ci][1][1] = (c[0]*c[8] - c[2]*c[6]) / dtrm;
		inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
		inverseCovs[ci][0][2] = (c[1]*c[5] - c[2]*c[4]) / dtrm;
		inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
		inverseCovs[ci][2][2] = (c[0]*c[4] - c[1]*c[3]) / dtrm;
	}		
}

/*
 * Calculate beta - parameter of GrabCut algorithm
 * beta = 1/ (2 *avg(sqr(||color[i]-color[j]||))
 */

static double calcBeta(const Mat & img) {
	double beta = 0.0;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			Vec3f color = img.at<Vec3f>(y,x);
			if (x > 0) { // node_i with node_i's LEFT 	
				Vec3f diff = color - img.at<Vec3f>(y, x-1);
				beta += diff.dot(diff);
			}
			if (y > 0 && x > 0) { // upleft
				Vec3f diff = color - img.at<Vec3f>(y-1, x-1);
				beta += diff.dot(diff);
			}
			if (y > 0) { // up
				Vec3f diff = color - img.at<Vec3f>(y-1, x);
				beta += diff.dot(diff);
			}
			if (y > 0 && x < img.cols-1) { // upright
				Vec3f diff = color - img.at<Vec3f>(y-1, x+1);
				beta += diff.dot(diff);
			}
		}
	}

	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0.0;
	else 
		beta = 1.0 / (2 * beta / (4*img.cols*img.rows - 3*(img.cols+img.rows) + 2));
	return beta;
}

/*
 * Calculate weights of noterminal vertices of graph
 * beta and gamma - parameters of grabcut algorithm
 */

static void calcNWeights(const Mat &img, Mat &leftW, Mat &upleftW, Mat &upW,
						Mat & uprightW, double beta, double gamma) {
	const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);

	leftW.create(img.rows, img.cols, CV_64FC1);
	upleftW.create(img.rows, img.cols, CV_64FC1);
	upW.create(img.rows, img.cols, CV_64FC1);
	uprightW.create(img.rows, img.cols, CV_64FC1);

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			Vec3f color = img.at<Vec3f>(y,x);

			if (x > 0) { // left
				Vec3f diff = color - img.at<Vec3f>(y, x-1);
				leftW.at<double>(y,x) = gamma*exp(-beta*diff.dot(diff));
			} else
				leftW.at<double>(y,x) = 0.0;

			if (x > 0 && y > 0) { // upleft
				Vec3f diff = color - img.at<Vec3f>(y-1,x-1);
				upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
			} else
				upleftW.at<double>(y,x) = 0.0;
			if (y > 0) { // up 
				Vec3f diff = color - img.at<Vec3f>(y-1, x);
				upW.at<double>(y,x) = gamma*exp(-beta*diff.dot(diff));
			} else
				upW.at<double>(y,x) = 0.0;

			if (x < img.cols - 1 && y > 0) { // upright
				Vec3f diff = color - img.at<Vec3f>(y-1,x+1);
				uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
			} else
				uprightW.at<double>(y,x) = 0.0;
		}
	}
}

/*
 * Check size, type and element values of mask matrix.
 */
static void checkMask(const Mat & img, const Mat & mask) {
	if (mask.empty()) {
		std::cerr<<"mask is empty"<<std::endl;
		return ;
	}
	if (mask.type() != CV_8UC1) {
		std::cerr<<"mask must have CV_8UC1 type"<<std::endl;
		return ;
	}
	if (mask.cols != img.cols || mask.rows != img.rows) {
		std::cerr<<"mask must have as many rows and cols as img"<<std::endl;
		return ;
	}

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			uchar val = mask.at<uchar>(y,x);
			if (val != _BGD && val != _FGD && val != _PR_BGD && val != _PR_FGD)	{
				std::cerr<<"mask has some invalid values"<<std::endl;
				return;
			}
		}
	}
}

/*
 * Initialize mask using rectangular.
 */
static void initMaskWithRect(Mat & mask, Size imgSize, Rect rect) {
	mask.create(imgSize, CV_8UC1);
	mask.setTo(_BGD);
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, imgSize.width - rect.x);
	rect.height = min(rect.height, imgSize.height - rect.y);
	(mask(rect)).setTo(_PR_FGD);	
}

/*
 * Initialize Gmm background and foreground models using kmeans algorithm.
 */
static void initGmms(const Mat &img, const Mat &mask, Gmm &bgdGmm, Gmm &fgdGmm) {
	const int kMeansItCount =  10;
	const int kMeansType = KMEANS_PP_CENTERS;

	Mat bgdLabels, fgdLabels;
	std::vector<Vec3f> bgdSamples, fgdSamples;
	Point p;

	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			if (mask.at<uchar>(p) == _BGD || mask.at<uchar>(p) == _PR_BGD) {
				bgdSamples.push_back((Vec3f)img.at<Vec3f>(p));
			} else {
				fgdSamples.push_back((Vec3f)img.at<Vec3f>(p));
			}
		}
	}
	assert(!fgdSamples.empty() && !bgdSamples.empty());

	// Using kMeans clustering bgd/fgd-Samples into componentCount clouters.
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSamples, Gmm::componentCount, bgdLabels, 
			TermCriteria(TermCriteria::COUNT, kMeansItCount, 0.0), 0, kMeansType);

	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, Gmm::componentCount, fgdLabels, 
			TermCriteria(TermCriteria::COUNT, kMeansItCount, 0.0), 0, kMeansType);
	
	/* Learing Gmm models */
	bgdGmm.initLearning();
	for (int i = 0; i < (int)bgdSamples.size(); i++) {
		bgdGmm.addSample(bgdLabels.at<int>(i), bgdSamples[i]);
	}
	bgdGmm.endLearning();

	fgdGmm.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++) {
		fgdGmm.addSample(fgdLabels.at<int>(i), fgdSamples[i]);
	}
	fgdGmm.endLearning();
}

// Iteration Alg: Step 1
/*
 * Assign Gmms components for each pixel.
 */
static void assignGmmsComponents(const Mat &img, const Mat &mask, const Gmm &bgdGmm,
		const Gmm &fgdGmm, Mat &comIdxs) {
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			Vec3d color = img.at<Vec3f>(p);
			comIdxs.at<int>(p) = mask.at<uchar>(p) == _BGD || mask.at<uchar>(p) == _PR_BGD ?
				bgdGmm.whichComponent(color) : fgdGmm.whichComponent(color);
		}
	}
}

// Iteration Alg: Step 2
/*
 * Learn Gmms parameters
 */
static void learnGmms(const Mat &img, const Mat &mask, const Mat &comIdxs, Gmm &bgdGmm, Gmm &fgdGmm) {
	Point p;

	bgdGmm.initLearning();
	fgdGmm.initLearning();
	for (int ci = 0; ci <= Gmm::componentCount; ci++) {
		for (p.y = 0; p.y < img.rows; p.y++) {
			for (p.x = 0; p.x < img.cols; p.x++) {
				if (comIdxs.at<int>(p) == ci) {
					if (mask.at<uchar>(p) == _BGD || mask.at<uchar>(p) == _PR_BGD) 
						bgdGmm.addSample(ci, img.at<Vec3f>(p));
					else
						fgdGmm.addSample(ci, img.at<Vec3f>(p));	
				}
			}
		}	
	}
	bgdGmm.endLearning();
	fgdGmm.endLearning();
}

/*
 * Construct GCGraph
 */
static void constructGCGraph(const Mat &img, const Mat &mask, const Gmm &bgdGmm, const Gmm &fgdGmm, double lambda, 
							const Mat &leftW, const Mat &upleftW, const Mat &upW, const Mat &uprightW, 
							GCGraph<double> &graph) {
	int vtxCount = img.rows * img.cols;
	int edgeCount = 2*(4*img.rows*img.cols - 3*(img.rows + img.cols) + 2);

	graph.create(vtxCount, edgeCount);
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			int vtxIdx = graph.addVtx();
			Vec3f color = img.at<Vec3f>(p);

			double fromSource, toSink;
			if (mask.at<uchar>(p) == _PR_FGD || mask.at<uchar>(p) == _PR_BGD) {
				fromSource = -log(fgdGmm(color)) * 5;
				toSink = -log(bgdGmm(color)) * 5;
			} else if (mask.at<uchar>(p) == _BGD) {
				fromSource = 0;
				toSink = lambda;
			} else {
				fromSource = lambda;
				toSink = 0;
			}
			graph.addTermWeights(vtxIdx, fromSource, toSink);

			if (p.x > 0) { // left
				double w = leftW.at<double>(p);
				graph.addEdges(vtxIdx, vtxIdx-1, w, w);
			}
			if (p.x > 0 && p.y > 0) { // upleft
				double w = upleftW.at<double>(p);
				graph.addEdges(vtxIdx-1, vtxIdx-img.cols-1, w, w);
			}
			if (p.y > 0) { // up
				double w = upW.at<double>(p);
				graph.addEdges(vtxIdx, vtxIdx-img.cols, w,w);
			}
			if (p.y > 0 && p.x < img.cols-1) { // upright
				double w = uprightW.at<double>(p);
				graph.addEdges(vtxIdx, vtxIdx - img.cols + 1, w, w);
			}
		}
	}

}

// Iteration Alg: Step 3
/*
 * Estimate segmentation using MaxFlow algorithm
 */
static void estimateSegmentation(GCGraph<double> &graph, Mat &mask) {
	graph.maxFlow();
	Point p;
	for (p.y = 0; p.y < mask.rows; p.y++) {
		for (p.x = 0; p.x < mask.cols; p.x++) {
			if (mask.at<uchar>(p) == _PR_BGD || mask.at<uchar>(p) == _PR_FGD) {
				if (graph.inSourceSegment(p.y * mask.cols + p.x)) // if TRUE: fgd
					mask.at<uchar>(p) = _PR_BGD;
				else
					mask.at<uchar>(p) = _PR_FGD;
			}
		}
	}
}

/* A comprehensive API */
void GrabCut(const Mat &img, Mat &mask, const Rect rect,  
		Mat &bgdModel, Mat &fgdModel, int iterCount, int mode) {
	if (img.empty()) {
		std::cerr<<"image is empty"<<std::endl;
		return ;
	}
	if (img.type() != CV_32FC3) {
		std::cerr<<"image must have CV_32FC3 type"<<std::endl;
		return ;
	}
	Gmm bgdGmm(bgdModel), fgdGmm(fgdModel);
	Mat compIdxs(img.size(), CV_32SC1);

	if (mode == _INIT_WITH_RECT || mode == _INIT_WITH_MASK) {
		if (mode == _INIT_WITH_RECT)
			initMaskWithRect(mask, img.size(), rect);
		else
			checkMask(img, mask);
		initGmms(img, mask, bgdGmm, fgdGmm);
	}
	if (iterCount <= 0)
		return ;
	if (mode == _EVAL)
		checkMask(img, mask);

	const double gamma = 50;
	const double lambda = 9*gamma;
	const double beta = calcBeta(img);

	Mat leftW, upleftW, upW, uprightW;
	calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
	for(int i = 0; i < iterCount; i++) {
		GCGraph<double> graph;
		assignGmmsComponents(img, mask, bgdGmm, fgdGmm, compIdxs);
		learnGmms(img, mask, compIdxs, bgdGmm, fgdGmm);
		constructGCGraph(img, mask, bgdGmm, fgdGmm, lambda, leftW, upleftW, upW, uprightW, graph);		
		estimateSegmentation(graph, mask);
	}
}
