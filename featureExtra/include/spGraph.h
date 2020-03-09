#ifndef __SPGRAPH_H__
#define __SPGRAPH_H__

#include "definition.h"
#include <set>
#include <iostream>
extern "C" {
#include "hog.h"
#include "lbp.h"
#include "generic.h"
#include "mathop.h"
};

class spGraph {
private:
	typedef struct Nodes {
		Nodes() : mean_color_(0.f, 0.f, 0.f), mean_hsv_(0.f, 0.f, 0.f), mean_lab_(0.f, 0.f, 0.f),
			mean_position_(0.f, 0.f), salient(false)
		{
			id_ = 0;
			size_ = 0;
			next_ = -1;
			perimeter_ = 0;
			background_ = 0;
		};
		int id_;
		int next_;
		int size_;
		int perimeter_;
		int background_;
		Vec3f mean_color_;
		Vec3f mean_hsv_;
		Vec3f mean_lab_;
		Vec2f mean_position_;
		set<int> neighbours_;
		bool salient;
	}Nodes;

	int n_;
	vector<Nodes> nodes;

	Vec3f mean_color_g_;
	Vec3f mean_hsv_g_;
	Vec3f mean_lab_g_;
	
	Vec2f mean_position_g_;
	double *geodesicMatrix;
	double *euclideanMatrix;

public:
	spGraph() 
		: mean_color_g_(0.f, 0.f, 0.f), mean_hsv_g_(0.f, 0.f, 0.f), mean_lab_g_(0.f, 0.f, 0.f),
			mean_position_g_(0.f, 0.f), geodesicMatrix(NULL), euclideanMatrix(NULL)
	{
	};

	spGraph(const Mat_<int> &seg_img, const Mat_<Vec3f> &color_img,
		const Mat_<Vec3f> &lab_img, const Mat_<Vec3f> &hsv_img)
		: mean_color_g_(0.f, 0.f, 0.f), mean_hsv_g_(0.f, 0.f, 0.f), mean_lab_g_(0.f, 0.f, 0.f),
			mean_position_g_(0.f, 0.f), geodesicMatrix(NULL), euclideanMatrix(NULL) 
	{
		spGraphcontruct(seg_img, color_img, lab_img, hsv_img);
	}
	~spGraph() {
		graphClear();
	};

	void graphClear() {
		mean_color_g_ = Vec3f(0.f, 0.f, 0.f);
		mean_hsv_g_ = Vec3f(0.f, 0.f, 0.f);
		mean_lab_g_ = Vec3f(0.f, 0.f, 0.f);
		mean_position_g_ = Vec2f(0.f, 0.f);
		if (geodesicMatrix != NULL) {
			delete geodesicMatrix;
			geodesicMatrix = NULL;
		}

		if (euclideanMatrix != NULL) {
			delete euclideanMatrix;
			euclideanMatrix = NULL;
		}

		nodes.clear();
		n_ = 0;
	}
	
	void spGraphcontruct(const Mat_<int> &seg_img, const Mat_<Vec3f> &color_img,
		const Mat_<Vec3f> &lab_img, const Mat_<Vec3f> &hsv_img);

	void gen_Kernel(const vector<float> &labels, Mat_<float> &kernel);
	void gen_Kernel(const Mat_<int>& labels, Mat_<float>& kernel);
public:
	int size() const {
		return n_;
	}
	int region_size(const int i) const {
		return nodes[i].size_;
	}
	double geoDist(int i, int j) const {
		assert(i >= 0 && i < n_ && j >= 0 && j < n_);
		return geodesicMatrix[i * n_ + j];
	}
	double eucliDist(int i, int j) const {
		assert(i >= 0 && i < n_ && j >= 0 && j < n_);
		return euclideanMatrix[i * n_ + j];
	}
	int backGroundness(int i) const {
		assert(i >= 0 && i < n_);
		return nodes[i].background_;
	}
	Vec3f mean_color(int i) const {
		assert(i >= 0 && i < n_);
		return nodes[i].mean_color_;
	}
	Vec3f mean_lab(int i) const {
		assert(i >= 0 && i < n_);
		return nodes[i].mean_lab_;
	}
	Vec3f mean_hsv(int i) const {
		assert(i >= 0 && i < n_);
		return nodes[i].mean_hsv_;
	}
	Vec2f mean_position(int i) const {
		assert(i >= 0 && i < n_);
		return nodes[i].mean_position_;
	}

	Vec3f mean_color_g(void) const {
		return mean_color_g_;
	}
	Vec3f mean_lab_g(void) const {
		return mean_lab_g_;
	}
	Vec3f mean_hsv_g(void) const {
		return mean_hsv_g_;
	}
	Vec2f mean_position_g(void) const {
		return mean_position_g_;
	}

	int region_next(int i) const {
		return nodes[i].next_;
	}
	int region_perimeter(int i) const {
		return nodes[i].perimeter_;
	}
	set<int> region_neighbors(int i) const {
		return nodes[i].neighbours_;
	}


protected:
	void euclidean() {
		for (int i = 0; i < n_; ++i) {
			for (int j = 0; j < n_; ++j) {
				Vec2f diff = nodes[i].mean_position_ - nodes[j].mean_position_;
				euclideanMatrix[i * n_ + j] = (diff.dot(diff));
			}
		}
	}
	void geodesicFloyd() {
		for (int k = 0; k < n_; ++k) {
			for (int i = 0; i < n_; ++i) {
				for (int j = 0; j < n_; ++j) {
					geodesicMatrix[i * n_ + j] = std::min(geodesicMatrix[i * n_ + k] + geodesicMatrix[k * n_ + j],
						geodesicMatrix[i * n_ + j]);
				}
			}
		}
		for (int i = 0; i < n_; ++i) {
			double minDist = DBL_MAX;
			for (int j = 0; j < n_; ++j) {
				if (minDist > geodesicMatrix[i * n_ + j]) {
					minDist = geodesicMatrix[i * n_ + j];
					nodes[i].next_ = j;
				}
			}
		}
	};
};

#endif