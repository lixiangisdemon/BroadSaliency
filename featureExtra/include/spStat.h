#ifndef _SPSTAT_H__
#define _SPSTAT_H__

#include "definition.h"
#include <set>
#include <iostream>
extern "C" {
#include "hog.h"
#include "lbp.h"
#include "generic.h"
#include "mathop.h"
};
#include "spGraph.h"

constexpr int LBP_DIMS = 0;
constexpr int HOG_DIMS = 31;
constexpr int COLOR_DIMS = 24;
constexpr int HSV_DIMS = 24;
constexpr int LAB_DIMS = 24;
constexpr int SPARSE_DIM = 24;
constexpr int TEXTURE_END = HOG_DIMS + COLOR_DIMS + HSV_DIMS + LAB_DIMS + LBP_DIMS;
template <int N>
struct floatVec_ {
	float data[N + 1];
	size_t size_;
	floatVec_() : size_(N) {
		memset(data, 0, sizeof(float) * size_);
	};
	floatVec_(const float f) : size_(N) {
		fill(f);
	};
	void norm() {
		float sum = 0;
		for (int i = 0; i < size_; ++i)
			sum += data[i];
		if (sum == 0.0)
			return;
		for (int i = 0; i < size_; ++i)
			data[i] /= (sum);
	}
	void fill(const float f) {
		if (f == 0)
			memset(data, 0, sizeof(float) * size_);
		else {
			for (int i = 0; i < size_; ++i)
				data[i] = f;
		}
	}
	size_t size() const {
		return size_;
	}
	float & operator [](const int i) {
		assert(i < size_ && i >= 0);
		return data[i];
	}
	float operator [](const int i) const {
		assert(i < size_ && i >= 0);
		return data[i];
	}
	void operator =(const float f) {
		fill(f);
	}
	operator float *() {
		return data;
	}
	floatVec_<N> operator + (const floatVec_<N> &a) const {
		floatVec_<N> c;
		for (int i = 0; i < size_; ++i)
			c[i] = data[i] + a[i];
		return c;
	}
	floatVec_<N> operator * (const float b) const {
		floatVec_<N> c;
		for (int i = 0; i < size_; ++i)
			c[i] = data[i] * b;
		return c;
	}
	floatVec_<N> operator / (const float b) const {
		floatVec_<N> c;
		for (int i = 0; i < size_; ++i)
			c[i] = data[i] / b;
		return c;
	}
	void operator += (const floatVec_<N> &a) {
		for (int i = 0; i < size_; ++i)
			data[i] += a[i];
	}
	void operator *= (const float b) {
		for (int i = 0; i < size_; ++i)
			data[i] *= b;
	}
	void operator /= (const float b) {
		for (int i = 0; i < size_; ++i)
			data[i] /= b;
	}
};

class spStat{
public:
	enum CONTRAST {
		RGB_CTR,
		HSV_CTR,
		LAB_CTR,
	};
	enum VARIANCE {
		RGB_VAR,
		HSV_VAR,
		LAB_VAR,
		SPATIAL_VAR,
	};
	enum BACKGROUND {
		RGB,
		HSV,
		LAB,
	};
private:
	float maxBoundary;
	float coeff_exp;
	string name_;
	spGraph graph_;
public:

	typedef floatVec_<HOG_DIMS> HOG_Vec;
	typedef vector<HOG_Vec> HOG_Vecs;
	typedef floatVec_<LBP_DIMS> LBP_Vec;
	typedef vector<LBP_Vec> LBP_Vecs;

	typedef floatVec_<COLOR_DIMS> colorHist;
	typedef vector<colorHist> colorHists;
	typedef floatVec_<HSV_DIMS> hsvHist;
	typedef vector<hsvHist> hsvHists;
	typedef floatVec_<LAB_DIMS> labHist;
	typedef vector<labHist> labHists;

	spStat() : maxBoundary(400), hogArray(NULL), lbpArray(NULL), cellSize(8)
	{ 
		coeff_exp = -8;
		name_ = "spStat";
		graph_.graphClear();
		stageChange(MYSTAGE::ATTACH);
	};
	~spStat() {
		clear();
		stageChange(MYSTAGE::THEREAD_DETACH);
		stageChange(MYSTAGE::DETACH);  
	};
	void clear() { 
		if (hogArray != NULL) { free(hogArray); hogArray = NULL; }
		if (lbpArray != NULL) { free(lbpArray); lbpArray = NULL; }
		graph_.graphClear();
	};

	void vecs_HogLbp(HOG_Vecs &hogVecs, LBP_Vecs &lbpVecs, const Mat_<int> &seg_img);
	colorHists rgbHists(const Mat_<int>& seg_img, const Mat_<Vec3f> & rgb, const spGraph &graph) const;
	hsvHists HSVHists(const Mat_<int>& seg_img, const Mat_<Vec3f>& hsv, const spGraph & graph) const;
	labHists LABHists(const Mat_<int>& seg_img, const Mat_<Vec3f>& lab, const spGraph & graph) const;

	void ctrs_HogLbp(vector<float> &hogCtrs, vector<float> &lbpCtrs,
			HOG_Vecs &hogVecs, LBP_Vecs &lbpVecs, const spGraph &graph_);
	void vars_HogLbp(vector<float>& hogCtrs, vector<float>& lbpCtrs, 
			HOG_Vecs& hogVecs, LBP_Vecs& lbpVecs, const spGraph & graph);
	void ctrs_color_hists(vector<float>& rbgCtrs, vector<float>& hsvCtrs, vector<float>& labCtrs, 
			spStat::colorHists & rgb_hists, spStat::hsvHists & hsv_hists, spStat::labHists & lab_hists, 
			const spGraph & graph);
	void vars_color_hists(vector<float>& rgbCtrs, vector<float>& hsvCtrs, vector<float>& labCtrs, 
			spStat::colorHists & rgb_hists, spStat::hsvHists & hsv_hists, spStat::labHists & lab_hists, 
			const spGraph & graph);

	void textures(const Mat & img);
	void hogFeatures(const float * image, int cols, int rows);
	void lbpFeatures(float * image, int cols, int rows);

	vector<float> backgroundness(const spGraph & graph, colorHists & hists);
	vector<float> backgroundness(const spGraph &graph);
	vector<float> objectness(const Mat_<int>& seg_img, const Mat_<Vec3f>& img, const spGraph &graph);
	void extractFeatures(const Mat & img, const Mat & gt_, Mat_<int>& seg_img, Mat_<float>& feats, vector<float>& labels);
	void distFeatures(const Mat & img, const Mat & gt_, Mat_<int>& lbls, Mat_<float> &feats1, Mat_<float> &feats2, 
		vector<float> &dists);

	vector<float> nodeContrast(const spGraph & graph, int flag) const;
	vector<float> nodeVariance(const spGraph & graph, int flag) const;
	Mat testing(const Mat & img);
	static void gen_labels(vector<float>& labels, const Mat_<float>& res, const Mat_<int>& seg_img);
	static void gen_map(Mat &res, const vector<float>& labels, const Mat_<int> &seg_img);
	pair<Mat_<float>, vector<float>> interpolating(const Mat_<float>& features, const vector<float>& labels, int size = 20);
	void backproject(const Mat_<float>& features, const vector<float>& nlabels, vector<float>& labels, int size = 20);

private:
	float *hogArray;
	float *lbpArray;
	int hogWidth;
	int hogHeight;
	int hogDimension;
	int cellSize; //defualt is 8
	int lbpHeight;
	int lbpWidth;
	int lbpDimension;
};
template <int N>
ostream & operator << (ostream &out, const floatVec_<N> &fv) {
	out << "[";
	for (int i = 0; i < fv.size_; ++i) {
		if (i < fv.size_ - 1)
			out << fv.data[i] << ", ";
		else
			out << fv.data[i];
	}
	out << "]";
	return out;
}

#endif