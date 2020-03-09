#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

#ifndef GRABCUT_H_
#define GRABCUT_H_

using namespace cv;

enum GC {
	_BGD,
	_FGD,
	_PR_BGD,
	_PR_FGD,
};

enum MODE {
	_INIT_WITH_RECT=10,
	_INIT_WITH_MASK,
	_WITHOUT_INIT,
	_EVAL,
};

class Gmm;

void GrabCut(const Mat &img, Mat &mask, const Rect rect,
	       	Mat &bgdModel, Mat &fgdModel, int iterCount = 1, int mode = _INIT_WITH_MASK);

#endif
