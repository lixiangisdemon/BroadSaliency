#include "spGraph.h"
#include <map>

void spGraph::spGraphcontruct(const Mat_<int> &seg_img, const Mat_<Vec3f> &color_img,
	const Mat_<Vec3f> &lab_img, const Mat_<Vec3f> &hsv_img) {

	graphClear();

	for (int j = 0; j < seg_img.rows; ++j) {
		for (int i = 0; i < seg_img.cols; ++i) {
			int id = seg_img(j, i);
			if (n_ <= id) {
				n_ = id + 1;
			}
		}
	}

	nodes.resize(n_);
	float max_bound = std::max(seg_img.rows, seg_img.cols);
	for (int i = 0; i < seg_img.rows; ++i) {
		for (int j = 0; j < seg_img.cols; ++j) {
			int id = seg_img(i, j);
			float x = (float)j;
			float y = (float)i;
			nodes[id].id_ = id;
			nodes[id].mean_position_ += Vec2f(x / seg_img.cols, y / seg_img.rows);
			nodes[id].mean_color_ += color_img(i, j);
			nodes[id].mean_hsv_ += hsv_img(i, j);
			nodes[id].mean_lab_ += lab_img(i, j);
			nodes[id].size_++;
			if (j > 0) {
				int left = seg_img(i, j - 1);
				if (id != left) {
					nodes[id].next_ = left;
					nodes[id].neighbours_.insert(left);
					nodes[left].next_ = id;
					nodes[left].neighbours_.insert(id);
				}
			}
			if (i > 0) {
				int up = seg_img(i - 1, j);
				if (id != up) {
					nodes[id].next_ = up;
					nodes[id].neighbours_.insert(up);
					nodes[up].next_ = id;
					nodes[up].neighbours_.insert(id);
				}
			}
			if (j > 0 && i > 0) {
				int upleft = seg_img(i - 1, j - 1);
				if (id != upleft) {
					nodes[id].next_ = upleft;
					nodes[id].neighbours_.insert(upleft);
					nodes[upleft].next_ = id;
					nodes[upleft].neighbours_.insert(id);
				}
			}
			if (i > 0 && j < seg_img.cols - 1) {
				int upright = seg_img(i - 1, j + 1);
				if (id != upright) {
					nodes[id].next_ = upright;
					nodes[id].neighbours_.insert(upright);
					nodes[upright].next_ = id;
					nodes[upright].neighbours_.insert(id);
				}
			}
			if (i == 0 || j == 0 || i == seg_img.rows - 1 || j == seg_img.cols - 1) {
				nodes[id].background_ = 1;
			}
			nodes[id].perimeter_ += (
				((j > 0) && (seg_img(i, j - 1) != id)) ||
				((i > 0) && (seg_img(i - 1, j) != id)) ||
				((j < seg_img.cols - 1) && (seg_img(i, j + 1) != id)) ||
				((i < seg_img.rows - 1) && (seg_img(i + 1, j) != id))
				);
		}
	}

	for (int i = 0; i < nodes.size(); ++i) {
		nodes[i].mean_position_ /= (nodes[i].size_ + 1e-20);
		nodes[i].mean_color_ /= (nodes[i].size_ + 1e-20);
		nodes[i].mean_hsv_ /= (nodes[i].size_ + 1e-20);
		nodes[i].mean_lab_ /= (nodes[i].size_ + 1e-20);
	}

	for (int i = 0; i < nodes.size(); ++i) {
		mean_position_g_ += nodes[i].mean_position_ / n_;
		mean_color_g_ += nodes[i].mean_color_ / n_;
		mean_hsv_g_ += nodes[i].mean_hsv_ / n_;
		mean_lab_g_ += nodes[i].mean_lab_ / n_;
	}
	geodesicMatrix = new double[n_ * n_];
	euclideanMatrix = new double[n_ * n_];

	euclidean();
	for (int i = 0; i < n_; ++i) {
		for (int j = 0; j < n_; ++j) {
			if (i == j) {
				geodesicMatrix[i * n_ + j] = 0;
			}
			else {
				if (nodes[i].neighbours_.find(j) == nodes[i].neighbours_.end()) {
					geodesicMatrix[i * n_ + j] = DBL_MAX;
				}
				else {
					geodesicMatrix[i * n_ + j] = eucliDist(i, j);
				}
			}

		}
	}
	geodesicFloyd();
};

void spGraph::gen_Kernel(const vector<float> &labels, Mat_<float> &kernel) {
	int n = labels.size();
	assert(n == n_);
	for (int i = 0; i < labels.size(); ++i) {
		nodes[i].salient = (labels[i] > 0.5 ? true : false);
	}
	float max_ = 0;
	kernel.create(n, n);
	for (int i = 0; i < n; ++i) {
		for (int j = i; j < n; ++j) {
			float diff_1 = eucliDist(i, j);
			Vec3f color_diff = (mean_color(i) - mean_color(j)) / 255.;
			float diff_2 = color_diff.dot(color_diff);
			int C = (nodes[i].salient != nodes[j].salient);
			kernel(i, j) = 8 * (diff_1 + diff_2 + 6 * C);
			kernel(j, i) = kernel(i, j);
		}
	}

	normalize(kernel, kernel, 0, 1, NORM_MINMAX);
}

void spGraph::gen_Kernel(const Mat_<int> &labels, Mat_<float> &kernel) {
	int n = labels.rows;
	assert(n == n_);
	for (int i = 0; i < labels.rows; ++i) {
		nodes[i].salient = (labels(i, 1) == 1 ? true : false);
	}
	float max_ = 0;
	kernel.create(n, n);
	for (int i = 0; i < n; ++i) {
		for (int j = i; j < n; ++j) {
			float diff_1 = eucliDist(i, j);
			Vec3f color_diff = (mean_color(i) - mean_color(j)) / 255.;
			float diff_2 = color_diff.dot(color_diff);
			int C = (nodes[i].salient != nodes[j].salient);
			kernel(i, j) = 8 * (diff_1 + diff_2 + 6 * C);
			kernel(j, i) = kernel(i, j);
		}
	}

	normalize(kernel, kernel, 0, 1, NORM_MINMAX);
}
