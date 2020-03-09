#ifndef __BLSCUT_H__
#define __BLSCUT_H__
#include "definition.h"
#include "spGraph.h"
#include "graph.hpp"
class BlsCut {
public:
	BlsCut() : coeff_(1) {};
	
	Mat blscut(const Mat& img, const Mat & sal, const Mat_<int> &seg_img, const Mat & kernel, const spGraph & graph_);

private:
	GCGraph<float> gc_graph_;
	float coeff_;
};
#endif