#include "blsCut.h"

Mat BlsCut::blscut(const Mat& img, const Mat & sal, const Mat_<int> &seg_img, 
	const Mat & Kernel, const spGraph & graph_)
{
	Mat rgb = img.clone();
	rgb.convertTo(rgb, CV_32F);
	Mat prior = sal.clone();
	Mat K = Kernel.clone();
	prior.convertTo(prior, CV_32F);
	K.convertTo(K, CV_32F);
	normalize(prior, prior, 0, 1, NORM_MINMAX);
	normalize(K, K, 0, 1, NORM_MINMAX);

	int num_edges = prior.size().area() - 3 * (prior.cols + prior.rows) + 2;
	gc_graph_.create(prior.size().area(), num_edges);
	Mat_<float> Left(prior.size()), Up(prior.size()), Up_left(prior.size()), Up_right(prior.size());
	double beta = 0;

	for (int i = 0; i < prior.rows; ++i) {
		for (int j = 0; j < prior.cols; ++j) {
			Point p(j, i);
			//cout << p << endl;
			float w = 0;
			// left
			if (j > 0) {
				Point left_(j - 1, i);
				int id1_ = seg_img(left_);
				int id2_ = seg_img(p);
				Vec3f diff = rgb.at<Vec3f>(p) - rgb.at<Vec3f>(left_);
				Vec2f diff2 = (Vec2f(j, i) - Vec2f(j - 1, i)) / 400;
				w = (diff.dot(diff) / 440 + diff2.dot(diff2));
				if (id1_ != id2_) {
					Vec3f diff_ = graph_.mean_color(id1_) - graph_.mean_color(id2_);
					Vec2f diff2_ = graph_.mean_position(id1_) - graph_.mean_position(id2_);
					float w_ = (diff_.dot(diff_) / 440 + diff2_.dot(diff2_));
					w = w / (w_ + 1e-20) * K.at<float>(id1_, id2_)*0.5 + 0.3*w;
				}
				beta += w;
				Left(p) = w;
			}

			// up
			if (i > 0) {
				Point up_(j, i - 1);
				int id1_ = seg_img(up_);
				int id2_ = seg_img(p);
				Vec3f diff = rgb.at<Vec3f>(p) - rgb.at<Vec3f>(up_);
				Vec2f diff2 = (Vec2f(j, i) - Vec2f(j, i - 1)) / 400;
				w = (diff.dot(diff) / 440 + diff2.dot(diff2));
				if (id1_ != id2_) {
					Vec3f diff_ = graph_.mean_color(id1_) - graph_.mean_color(id2_);
					Vec2f diff2_ = graph_.mean_position(id1_) - graph_.mean_position(id2_);
					float w_ = (diff_.dot(diff_) / 440 + diff2_.dot(diff2_));
					w = w / (w_ + 1e-20) * K.at<float>(id1_, id2_)*0.5 + 0.3*w;
				}
				beta += w;
				Up(p) = w;
			}

			// Up_left
			if (i > 0 && j > 0) {
				Point up_left(j - 1, i - 1);
				int id1_ = seg_img(up_left);
				int id2_ = seg_img(p);
				Vec3f diff = rgb.at<Vec3f>(p) - rgb.at<Vec3f>(up_left);
				Vec2f diff2 = (Vec2f(j, i) - Vec2f(j, i - 1)) / 400;
				w = (diff.dot(diff) / 440 + diff2.dot(diff2));
				if (id1_ != id2_) {
					Vec3f diff_ = graph_.mean_color(id1_) - graph_.mean_color(id2_);
					Vec2f diff2_ = graph_.mean_position(id1_) - graph_.mean_position(id2_);
					float w_ = (diff_.dot(diff_) / 440 + diff2_.dot(diff2_));
					w = w / (w_ + 1e-20) * K.at<float>(id1_, id2_)*0.5 + 0.3*w;
				}
				beta += w;
				Up_left(p) = w;
			}

			// up_right
			if (i > 0 && j < prior.cols - 1) {
				Point up_right(j + 1, i - 1);
				int id1_ = seg_img(up_right);
				int id2_ = seg_img(p);
				Vec3f diff = rgb.at<Vec3f>(p) - rgb.at<Vec3f>(up_right);
				Vec2f diff2 = (Vec2f(j, i) - Vec2f(j - 1, i)) / 400;
				w = (diff.dot(diff) / 440 + diff2.dot(diff2));
				if (id1_ != id2_) {
					Vec3f diff_ = graph_.mean_color(id1_) - graph_.mean_color(id2_);
					Vec2f diff2_ = graph_.mean_position(id1_) - graph_.mean_position(id2_);
					float w_ = (diff_.dot(diff_) / 440 + diff2_.dot(diff2_));
					w = w / (w_ + 1e-20) * K.at<float>(id1_, id2_)*0.5 + 0.3*w;
				}
				beta += w;
				Up_right(p) = w;
			}
		}
	}
	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1. / (2 * beta / num_edges);

	for (int i = 0; i < prior.rows; ++i) {
		for (int j = 0; j < prior.cols; ++j) {
			Point p(j, i);
			float pos = prior.at<float>(i, j);
			float neg = 1 - prior.at<float>(i, j);
			neg = (neg > 0.5 && neg < 0.6) ? 0.6 : neg;
			float fromsource = -log(pos);
			float tosink = -log(neg);
			int count_ = gc_graph_.addVtx();

			gc_graph_.addTermWeights(count_, fromsource, tosink);
			
			// left
			if (j > 0) {
				double w = coeff_ * exp(-beta * Left(p));
				gc_graph_.addEdges(count_, count_ - 1, w, w);
			}

			// up
			if (i > 0) {
				double w = coeff_ * exp(-beta * Up(p));
				gc_graph_.addEdges(count_, count_ - prior.cols, w, w);
			}

			// Up_left
			if (j > 0 && i > 0) {
				double w = coeff_ * exp(-beta * Up_left(p));
				gc_graph_.addEdges(count_, count_ - 1 - prior.cols, w, w);
			}

			// up_right
			if (i > 0 && j < prior.cols - 1) {
				double w = coeff_ * exp(-beta * Up_right(p));
				gc_graph_.addEdges(count_, count_ - prior.cols + 1, w, w);
			}
		}
	}

	gc_graph_.maxFlow();
	Mat res(prior.size(), CV_32F);
	//vector<float> results(prior.size().area());
	for (int i = 0; i < res.rows; ++i) {
		for (int j = 0; j < res.cols; ++j) {
			int id = i * res.cols + j;
			if (gc_graph_.inSourceSegment(id)) {
				//results[id] = 0;
				res.at<float>(i, j) = 0;
			}
			else {
				//results[id] = 1;
				res.at<float>(i, j) = 1;
			}
		}

	}
	Mat result;
	cv::GaussianBlur(res, result, Size(3, 3), 2, 2);
	return result;
}