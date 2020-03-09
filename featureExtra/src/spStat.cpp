#include "spStat.h"
#include <set>
#include <random>
#include <functional>
#include "slic.h"
/* 
* @Params:
*	Inputs:
*		img, the image Mat that is going to preceed
*	Outputs:
*		feats, return a nLabels' feature vectors, and each vector is 101 dims. 
*			the feature vector is the features extracted from region in multi-view, and they are:
*				RGB-3dims, Color_Contrast-1dims, Location-2dims, Size-1dims, Spatial_Variance-1dims,
*				HOG-HOG_DIMSdims, HOG-Contrast-1dims, LBP-58dims, LBP_Contrast-1dims,
*				Objectness-1dims, Backgroundness-1dims.
*/

void spectralResidual(const Mat &image, Mat &saliencyMap, int scale_1 = 128, int scale_2 = 128);

spStat::colorHists spStat::rgbHists(const Mat_<int> &seg_img, const Mat_<Vec3f> &Rgb, const spGraph &graph) const {
	int n = graph.size();
	colorHists hists(n);
	Mat_<Vec3f> rgb = Rgb / 255.;
	int channel_dims = COLOR_DIMS / 3;
	for (int i = 0; i < rgb.rows; ++i) {
		const float *r = rgb.ptr<float>(i);
		const float *g = rgb.ptr<float>(i) + 1;
		const float *b = rgb.ptr<float>(i) + 2;
		for (int j = 0; j < rgb.cols; ++j) {
			int id = seg_img(i, j);
			int r_id = *(r + j * 3) * (channel_dims);
			int g_id = *(g + j * 3) * (channel_dims) + channel_dims;
			int b_id = *(b + j * 3) * (channel_dims) + channel_dims * 2;
			r_id = min(channel_dims - 1, r_id);
			g_id = min(2 * channel_dims - 1, g_id);
			b_id = min(3 * channel_dims - 1, b_id);
			hists[id][r_id] += 1. / (graph.region_size(id) + 1e-20);
			hists[id][g_id] += 1. / (graph.region_size(id) + 1e-20);
			hists[id][b_id] += 1. / (graph.region_size(id) + 1e-20);
		}
	}
	return hists;
}

spStat::hsvHists spStat::HSVHists(const Mat_<int> &seg_img, const Mat_<Vec3f> &Hsv, const spGraph &graph) const {
	int n = graph.size();
	hsvHists hists(n);
	Mat_<Vec3f> hsv = Hsv / 255.;
	int channel_dims = HSV_DIMS / 2;
	int h_channel_dims = HSV_DIMS / 4;
	for (int i = 0; i < hsv.rows; ++i) {
		const float *h = hsv.ptr<float>(i);
		const float *s = hsv.ptr<float>(i) + 1;
		const float *v = hsv.ptr<float>(i) + 2;
		for (int j = 0; j < hsv.cols; ++j) {
			int id = seg_img(i, j);
			int h_id = *(h + j * 3) * (channel_dims);
			int s_id = *(s + j * 3) * (h_channel_dims) + channel_dims;
			int v_id = *(v + j * 3) * (h_channel_dims) + h_channel_dims * 3;
			h_id = min(channel_dims - 1, h_id);
			s_id = min(3 * h_channel_dims - 1, s_id);
			v_id = min(HSV_DIMS - 1, v_id);
			hists[id][h_id] += 1. / (graph.region_size(id) + 1e-20);
			hists[id][s_id] += 1. / (graph.region_size(id) + 1e-20);
			hists[id][v_id] += 1. / (graph.region_size(id) + 1e-20);
		}
	}
	return hists;
}

spStat::labHists spStat::LABHists(const Mat_<int> &seg_img, const Mat_<Vec3f> &Lab, const spGraph &graph) const {
	int n = graph.size();
	labHists hists(n);
	Mat_<Vec3f> lab = Lab / 255.;
	int channel_dims = LAB_DIMS / 2;
	int h_channel_dims = LAB_DIMS / 4;
	for (int i = 0; i < lab.rows; ++i) {
		const float *l = lab.ptr<float>(i);
		const float *a = lab.ptr<float>(i) + 1;
		const float *b = lab.ptr<float>(i) + 2;
		for (int j = 0; j < lab.cols; ++j) {
			int id = seg_img(i, j);
			int l_id = *(l + j * 3) * (channel_dims);
			int a_id = *(a + j * 3) * (h_channel_dims) + channel_dims;
			int b_id = *(b + j * 3) * (h_channel_dims) + h_channel_dims * 3;
			l_id = min(channel_dims - 1, l_id);
			a_id = min(3 * h_channel_dims - 1, a_id);
			b_id = min(LAB_DIMS - 1, b_id);
			hists[id][l_id] += 1. / (graph.region_size(id) + 1e-20);
			hists[id][a_id] += 1. / (graph.region_size(id) + 1e-20);
			hists[id][b_id] += 1. / (graph.region_size(id) + 1e-20);
		}
	}
	return hists;
}

void spStat::ctrs_HogLbp(vector<float>& hogCtrs, vector<float>& lbpCtrs, 
	HOG_Vecs& hogVecs, LBP_Vecs& lbpVecs,
	const spGraph &graph)
{
	int regions = graph.size();
	VlFloatVectorComparisonFunction dist_chi = vl_get_vector_comparison_function_f(VlVectorComparisonType::VlDistanceChi2);
	if (hogArray != NULL) {
		hogCtrs.resize(regions);
//#pragma omp parallel for
		for (int i = 0; i < regions; ++i) {
			for (int j = 0; j < regions; ++j) {
				float exp_term = exp(coeff_exp * graph.eucliDist(i, j));
				hogCtrs[i] += dist_chi(HOG_DIMS, (float *)hogVecs[i], (float *)hogVecs[j]) *
					exp_term;
			}
		}
		normalize(hogCtrs, hogCtrs, 0, 1, NORM_MINMAX);
	}
	
	if (lbpArray) {
		lbpCtrs.resize(regions);
//#pragma omp parallel for
		for (int i = 0; i < regions; ++i) {
			for (int j = 0; j < regions; ++j) {
				float exp_term = exp(coeff_exp * graph.eucliDist(i, j));
				lbpCtrs[i] += dist_chi(LBP_DIMS, (float *)lbpVecs[i], (float *)lbpVecs[j]) *
					exp_term;
			}
		}
		normalize(lbpCtrs, lbpCtrs, 0, 1, NORM_MINMAX);
	}
}

void spStat::vars_HogLbp(vector<float>& hogVars, vector<float>& lbpVars,
	HOG_Vecs& hogVecs, LBP_Vecs& lbpVecs,
	const spGraph &graph)
{
	int regions = graph.size();
	VlFloatVectorComparisonFunction dist_chi = vl_get_vector_comparison_function_f(VlVectorComparisonType::VlDistanceChi2);
	
	if (hogArray != NULL) {
		hogVars.resize(regions);
		HOG_Vec mean_hog;
		for (int i = 0; i < regions; ++i) {
			mean_hog += hogVecs[i] * (1. / regions);
		}
//#pragma omp parallel for
		for (int i = 0; i < regions; ++i) {
			hogVars[i] = dist_chi(HOG_DIMS, (float *)hogVecs[i], (float *)mean_hog);

		}
		normalize(hogVars, hogVars, 0, 1, NORM_MINMAX);
	}

	if (lbpArray != NULL) {
		lbpVars.resize(regions);
		LBP_Vec mean_lbp;
		for (int i = 0; i < regions; ++i) {
			mean_lbp += lbpVecs[i] * (1. / regions);
		}
//#pragma omp parallel for
		for (int i = 0; i < regions; ++i) {
			lbpVars[i] = dist_chi(LBP_DIMS, (float *)lbpVecs[i], (float *)mean_lbp);
		}
		normalize(lbpVars, lbpVars, 0, 1, NORM_MINMAX);
	}
}

void spStat::ctrs_color_hists(vector<float>& rgbCtrs, vector<float>& hsvCtrs, vector<float>& labCtrs,
	spStat::colorHists& rgb_hists, spStat::hsvHists& hsv_hists, spStat::labHists& lab_hists,
	const spGraph &graph)
{
	int regions = graph.size();
	rgbCtrs.resize(regions);
	hsvCtrs.resize(regions);
	labCtrs.resize(regions);
	VlFloatVectorComparisonFunction dist_chi = vl_get_vector_comparison_function_f(VlVectorComparisonType::VlDistanceChi2);
//#pragma omp parallel for
	for (int i = 0; i < regions; ++i) {
		float sum = 0.;
		for (int j = 0; j < regions; ++j) {
			float exp_term = exp(coeff_exp * graph.eucliDist(i, j));
			rgbCtrs[i] += dist_chi(COLOR_DIMS, (float *)rgb_hists[i], (float *)rgb_hists[j]) *
				exp_term;
			hsvCtrs[i] += dist_chi(HSV_DIMS, (float *)hsv_hists[i], (float *)hsv_hists[j]) *
				exp_term;
			labCtrs[i] += dist_chi(LAB_DIMS, (float *)lab_hists[i], (float *)lab_hists[j]) *
				exp_term;
			sum += exp_term;
		}
	}
	normalize(rgbCtrs, rgbCtrs, 0, 1, NORM_MINMAX);
	normalize(hsvCtrs, hsvCtrs, 0, 1, NORM_MINMAX);
	normalize(labCtrs, labCtrs, 0, 1, NORM_MINMAX);
}

void spStat::vars_color_hists(vector<float>& rgbVars, vector<float>& hsvVars, vector<float>& labVars,
	spStat::colorHists& rgb_hists, spStat::hsvHists& hsv_hists, spStat::labHists& lab_hists,
	const spGraph &graph)
{
	int regions = graph.size();
	rgbVars.resize(regions);
	hsvVars.resize(regions);
	labVars.resize(regions);
	VlFloatVectorComparisonFunction dist_chi = vl_get_vector_comparison_function_f(VlVectorComparisonType::VlDistanceL2);
	colorHist mean_rgb;
	hsvHist mean_hsv;
	labHist mean_lab;
	for (int i = 0; i < regions; ++i) {
		mean_rgb += rgb_hists[i] * (1. / regions);
		mean_hsv += hsv_hists[i] * (1. / regions);
		mean_lab += lab_hists[i] * (1. / regions);
	}
//#pragma omp parallel for
	for (int i = 0; i < regions; ++i) {
		rgbVars[i] = dist_chi(COLOR_DIMS, (float *)rgb_hists[i], (float *)mean_rgb);
		hsvVars[i] = dist_chi(HSV_DIMS, (float *)hsv_hists[i], (float *)mean_hsv);
		labVars[i] = dist_chi(LAB_DIMS, (float *)lab_hists[i], (float *)mean_lab);
	}
	normalize(rgbVars, rgbVars, 0, 1, NORM_MINMAX);
	normalize(hsvVars, hsvVars, 0, 1, NORM_MINMAX);
	normalize(labVars, labVars, 0, 1, NORM_MINMAX);
}

void spStat::vecs_HogLbp(HOG_Vecs &hogVecs,
	LBP_Vecs &lbpVecs,
	const Mat_<int> &seg_img) {

	int rows = seg_img.rows;
	int cols = seg_img.cols;
	int nLabels = hogVecs.size();

	if (hogArray != NULL) {
		for (int i = 0; i < hogHeight; ++i) {
			for (int j = 0; j < hogWidth; ++j) {
				int u_bound = std::max(i * cellSize, 0);
				int b_bound = std::min((i + 1) * cellSize, rows);
				int l_bound = std::max(j * cellSize, 0);
				int r_bound = std::min((j + 1) * cellSize, cols);
				vector<int> counts(nLabels);
				for (int y = u_bound; y < b_bound; y++) {
					for (int x = l_bound; x < r_bound; x++) {
						if (++counts[seg_img(y, x)] == cellSize * cellSize / 4) {
							for (int k = 0; k < hogDimension; k++) {
								hogVecs[seg_img(y, x)][k] += hogArray[k * hogHeight * hogWidth + i * hogWidth + j];
							}
						}
					}
				}

			}
		}
		for (int i = 0; i < nLabels; ++i) {
			hogVecs[i].norm();
		}
	}
	if (lbpArray != NULL) {
		for (int i = 0; i < lbpHeight; ++i) {
			for (int j = 0; j < lbpWidth; ++j) {
				int u_bound = std::max(i * cellSize, 0);
				int b_bound = std::min((i + 1) * cellSize, rows);
				int l_bound = std::max(j * cellSize, 0);
				int r_bound = std::min((j + 1) * cellSize, cols);
				vector<int> counts(nLabels);
				for (int y = u_bound; y < b_bound; y++) {
					for (int x = l_bound; x < r_bound; x++) {
						if (++counts[seg_img(y, x)] == cellSize * cellSize / 4) {
							for (int k = 0; k < lbpDimension; k++) {
								lbpVecs[seg_img(y, x)][k] += lbpArray[k * lbpHeight * lbpWidth + i * lbpWidth + j];
							}
						}
					}
				}

			}
		}
		for (int i = 0; i < nLabels; ++i) {
			lbpVecs[i].norm();
		}
	}
}

void spStat::textures(const Mat &img) {
	Mat tmp = img.clone();
	Mat divided_channels[3];
	split(tmp, divided_channels);
	float *image = new float[img.rows * img.cols * 3];
	for (int i = 0; i < 3; ++i) {
		int offset = i * img.rows * img.cols;
		memcpy(image + offset, divided_channels[i].data, img.rows * img.cols * sizeof(float));
	}
	if (HOG_DIMS > 0)
		hogFeatures(image, img.cols, img.rows);
	if (LBP_DIMS > 0)
		lbpFeatures(image, img.cols, img.rows);

	delete [] image;
}

void spStat::hogFeatures(const float *image, int cols, int rows) {
	VlHog *hog = vl_hog_new(VlHogVariant::VlHogVariantUoctti, 9, false);
	vl_hog_put_image(hog, image, cols, rows, 3, cellSize);
	hogWidth = vl_hog_get_width(hog);
	hogHeight = vl_hog_get_height(hog);
	hogDimension = vl_hog_get_dimension(hog);
	hogArray = (float *)vl_malloc(hogWidth * hogHeight * hogDimension * sizeof(float));
	vl_hog_extract(hog, hogArray);

	vl_hog_delete(hog);
}

void spStat::lbpFeatures(float *image, int cols, int rows) {
	lbpHeight = rows / cellSize;
	lbpWidth = cols / cellSize;
	int cstride = lbpHeight * lbpWidth;
	VlLbp *lbp = vl_lbp_new(VlLbpMappingType::VlLbpUniform, false);
	lbpDimension = vl_lbp_get_dimension(lbp);
	lbpArray = (float *)vl_malloc(sizeof(float) * lbpDimension * cstride);
	vl_lbp_process(lbp, lbpArray, image, cols, rows, cellSize);
	vl_lbp_delete(lbp);
}

vector<float> spStat::backgroundness(const spGraph &graph, colorHists &hists) {
	vector<float> bks(graph.size());
	VlFloatVectorComparisonFunction dist_chi = vl_get_vector_comparison_function_f(VlVectorComparisonType::VlDistanceChi2);
//#pragma omp parallel for
	for (int i = 0; i < graph.size(); ++i) {
		bks[i] = 0;
		float sum = 0.0;
		for (int j = 0; j < graph.size(); ++j) {
			float dist = dist_chi(COLOR_DIMS, (float *)hists[i], (float *)hists[j]);
			float sp_dist = graph.eucliDist(i, j);
			bks[i] += std::exp(-1 * dist) * sp_dist;
		}
		//bks[i] /= sum;
	}
	normalize(bks, bks, 0, 1, NORM_MINMAX);
	return bks;
}

vector<float> spStat::backgroundness(const spGraph &graph) {
	vector<float> bks(graph.size());
	Vec3f diff_lab;
//#pragma omp parallel for
	for (int i = 0; i < graph.size(); ++i) {
		bks[i] = 0;
		float sum = 0.0;
		for (int j = 0; j < graph.size(); ++j) {
			diff_lab = graph.mean_lab(i) - graph.mean_lab(j);
			float lab_dist = diff_lab.dot(diff_lab);
			float sp_dist = graph.eucliDist(i, j);
			float exp_term = diff_lab.dot(diff_lab) / 411;
			bks[i] += std::exp(-0.5 * exp_term) * sp_dist;
			//sum += std::exp(-0.5 * exp_term);
		}
		//bks[i] /= sum;
	}
	normalize(bks, bks, 0, 1, NORM_MINMAX);
	return bks;
}

vector<float> spStat::objectness(const Mat_<int> &seg_img, const Mat_<Vec3f> &img, const spGraph &graph) {
	int n = graph.size();
	vector<float> objs(n);
	Mat sal_map;
	int scale_1 = 128;
	int scale_2 = 128;
	spectralResidual(img, sal_map, scale_1, scale_2);
	for (int i = 0; i < seg_img.rows; ++i) {
		for (int j = 0; j < seg_img.cols; ++j) {
			int id = seg_img(i, j);
			objs[id] += sal_map.at<float>(i, j) / graph.region_size(id);
		}
	}
	vector<float> objectnesses(n);
	fill(objectnesses.begin(), objectnesses.end(), 0);
	for (int i = 0; i < objs.size(); ++i) {
		for (int j = 0; j < objs.size(); ++j) {
			float dist_ij = graph.eucliDist(i, j);
			objectnesses[i] += dist_ij * objs[j];
		}
	}
	normalize(objectnesses, objectnesses, 0, 1, NORM_MINMAX);
	return objectnesses;
}

void spectralResidual(const Mat &image, Mat &saliencyMap, int scale_1, int scale_2) {
	Mat grayTemp, grayDown;
	vector<Mat> mv;
	
	Size resizedImageSize(scale_1, scale_2);
	Mat realImage(resizedImageSize, CV_64F);
	Mat imaginaryImage(resizedImageSize, CV_64F);
	imaginaryImage.setTo(0);
	Mat combinedImage(resizedImageSize, CV_64FC2);
	Mat imageDFT;
	Mat logAmplitude;
	Mat angle(resizedImageSize, CV_64F);
	Mat magnitude(resizedImageSize, CV_64F);
	Mat logAmplitude_blur, imageGR;

	if (image.channels() == 3) {
		cvtColor(image, imageGR, COLOR_BGR2GRAY);
		resize(imageGR, grayDown, resizedImageSize, 0, 0, INTER_CUBIC);

	}
	else {
		imageGR = image.clone();
		resize(imageGR, grayDown, resizedImageSize, 0, 0, INTER_CUBIC);
	}

	grayDown.convertTo(realImage, CV_64F);
	mv.push_back(realImage);
	mv.push_back(imaginaryImage);
	merge(mv, combinedImage);
	dft(combinedImage, imageDFT);
	split(imageDFT, mv);

	//-- Get magnitude and phase of frequency spectrum
	cartToPolar(mv.at(0), mv.at(1), magnitude, angle);
	log(magnitude + Scalar(1), logAmplitude);
	//-- Blur log amplitude with averaging filter
	blur(logAmplitude, logAmplitude_blur, Size(3, 3));
	exp(logAmplitude - logAmplitude_blur, magnitude);

	//-- Back to cartesian frequency domain
	polarToCart(magnitude, angle, mv[0], mv[1], false);
	merge(mv, imageDFT);
	dft(imageDFT, combinedImage, DFT_INVERSE);
	split(combinedImage, mv);
	cartToPolar(mv[0], mv[1], magnitude, angle, false);
	magnitude = magnitude.mul(magnitude);

	double minVal, maxVal;
	minMaxLoc(magnitude, &minVal, &maxVal);
	magnitude = magnitude / maxVal;
	magnitude.convertTo(magnitude, CV_32F);
	resize(magnitude, saliencyMap, image.size(), 0, 0, INTER_LINEAR);
	GaussianBlur(saliencyMap, saliencyMap, Size(3, 3), 1, 1);
}

void spStat::extractFeatures(const Mat & img, const Mat & gt_, Mat_<int>& seg_img, Mat_<float> &feats, vector<float> &labels)
{
	clear();
	Mat img_ = img.clone();
	// const float scale = maxBoundary / (std::max(img_.rows, img_.cols) + 1e-20);
	// resize(img_, img_, Size(img_.cols * scale, img_.rows * scale), 0, 0, INTER_LINEAR);
	// Mat temp = img_.clone();
	// bilateralFilter(temp, img_, 3, 40, 50);
	CV_Assert(img_.size().area() > 400);

	Mat LAB, HSV;
	cvtColor(img_, LAB, COLOR_RGB2Lab);
	cvtColor(img_, HSV, COLOR_RGB2HSV);
	Mat_<Vec3f> rgb, lab, hsv;
	img_.convertTo(rgb, CV_32FC3);
	LAB.convertTo(lab, CV_32FC3);
	HSV.convertTo(hsv, CV_32FC3);

	Slic slic;
	seg_img = slic.generate_superpixels(LAB, -1, 40, 400);
	graph_.spGraphcontruct(seg_img, rgb, lab, hsv);
	int nLabels = graph_.size();

	vector<float> RGBcontrast = nodeVariance(graph_, VARIANCE::RGB_VAR);
	vector<float> HSVcontrast = nodeVariance(graph_, VARIANCE::HSV_VAR);
	vector<float> LABcontrast = nodeVariance(graph_, VARIANCE::LAB_VAR);
	vector<float> objs = nodeVariance(graph_, VARIANCE::SPATIAL_VAR);

	textures(rgb);
	HOG_Vecs hogVecs;
	LBP_Vecs lbpVecs;
	if (hogArray) {
		hogVecs.resize(nLabels);
	}
	if (lbpArray) {
		lbpVecs.resize(nLabels);
	}
	vecs_HogLbp(hogVecs, lbpVecs, seg_img);
	colorHists rgb_hists = rgbHists(seg_img, rgb, graph_);
	hsvHists hsv_hists = HSVHists(seg_img, hsv, graph_);
	labHists lab_hists = LABHists(seg_img, lab, graph_);
	vector<float> rgbCtrs, hsvCtrs, labCtrs;
	vars_color_hists(rgbCtrs, hsvCtrs, labCtrs, rgb_hists, hsv_hists, lab_hists, graph_);

	vector<float> bks = backgroundness(graph_);
	Mat_<float> texture(nLabels, TEXTURE_END);
	Mat_<float> color(nLabels, 9);
	Mat_<float> contra(nLabels, 27);	
	Mat_<float> loc(nLabels, 3);

	for (int i = 0; i < nLabels; ++i) {
		float *text_ptr = texture.ptr<float>(i);
		float *color_ptr = color.ptr<float>(i);
		float *loc_ptr = loc.ptr<float>(i);
		float *contra_ptr = contra.ptr<float>(i);

		/* Texture: */

		for (int j = 0; j < rgb_hists[i].size(); ++j) {
			*text_ptr++ = rgb_hists[i][j];
		}

		for (int j = 0; j < hsv_hists[i].size(); ++j) {
			*text_ptr++ = hsv_hists[i][j];
		}

		for (int j = 0; j < lab_hists[i].size(); ++j) {
			*text_ptr++ = lab_hists[i][j];
		}	
		for (int j = 0; i < hogVecs.size() && j < hogVecs[i].size(); ++j) {
			*text_ptr++ = hogVecs[i][j];
		}	
		for (int j = 0; i < lbpVecs.size() && j < lbpVecs[i].size(); ++j) {
			*text_ptr++ = lbpVecs[i][j];
		}

		/* Spatial: */
		Vec2f pos = graph_.mean_position(i);
		*loc_ptr++ = pos[0];
		*loc_ptr++ = pos[1];
		*loc_ptr++ = (float)graph_.region_size(i) / seg_img.size().area();

		/* Color: */
		Vec3f mean_color = graph_.mean_color(i);
		Vec3f mean_hsv = graph_.mean_hsv(i);
		Vec3f mean_lab = graph_.mean_lab(i);
		for (int j = 0; j < 3; ++j)
			*color_ptr++ = mean_color[j] / 255.;
		for (int j = 0; j < 3; ++j)
			*color_ptr++ = mean_hsv[j] / 255.;
		for (int j = 0; j < 3; ++j) 
			*color_ptr++ = mean_lab[j] / 255.;

		/* Contrast: */
		*contra_ptr++ = exp(-3 * objs[i]);
		*contra_ptr++ = exp(-3 * bks[i]);
		*contra_ptr++ = exp(-3 * bks[i]) * exp(-3 * objs[i]);
		*contra_ptr++ = RGBcontrast[i];
		*contra_ptr++ = HSVcontrast[i];
		*contra_ptr++ = LABcontrast[i];
		*contra_ptr++ = rgbCtrs[i];
		*contra_ptr++ = labCtrs[i];
		*contra_ptr++ = hsvCtrs[i];

		*contra_ptr++ = RGBcontrast[i] * exp(-3 * bks[i]);
		*contra_ptr++ = HSVcontrast[i] * exp(-3 * bks[i]);
		*contra_ptr++ = LABcontrast[i] * exp(-3 * bks[i]);
		*contra_ptr++ = rgbCtrs[i] * exp(-3 * bks[i]);
		*contra_ptr++ = labCtrs[i] * exp(-3 * bks[i]);
		*contra_ptr++ = hsvCtrs[i] * exp(-3 * bks[i]);

		*contra_ptr++ = RGBcontrast[i] * exp(-3 * objs[i]);
		*contra_ptr++ = HSVcontrast[i] * exp(-3 * objs[i]);
		*contra_ptr++ = LABcontrast[i] * exp(-3 * objs[i]);
		*contra_ptr++ = rgbCtrs[i] * exp(-3 * objs[i]);
		*contra_ptr++ = labCtrs[i] * exp(-3 * objs[i]);
		*contra_ptr++ = hsvCtrs[i] * exp(-3 * objs[i]);

		*contra_ptr++ = RGBcontrast[i] * exp(-3 * objs[i]) * exp(-3 * bks[i]);
		*contra_ptr++ = HSVcontrast[i] * exp(-3 * objs[i]) * exp(-3 * bks[i]);
		*contra_ptr++ = LABcontrast[i] * exp(-3 * objs[i]) * exp(-3 * bks[i]);
		*contra_ptr++ = rgbCtrs[i] * exp(-3 * objs[i]) * exp(-3 * bks[i]);
		*contra_ptr++ = labCtrs[i] * exp(-3 * objs[i]) * exp(-3 * bks[i]);
		*contra_ptr++ = hsvCtrs[i] * exp(-3 * objs[i]) * exp(-3 * bks[i]);

		// CV_Assert(text_ptr - texture.ptr<float>(i) == TEXTURE_END);
		// CV_Assert(color_ptr - color.ptr<float>(i) == 9);
		// CV_Assert(loc_ptr - loc.ptr<float>(i) == 3);
		// CV_Assert(contra_ptr - contra.ptr<float>(i) == 27);
	}

	for (int i = 0; i < contra.cols; ++i) {
		normalize(contra.col(i), contra.col(i), 0, 1, NORM_MINMAX);
	}
	Mat_<float> w_, u_, vt_;
	SVD::compute(texture, w_, u_, vt_);
	Mat_<float> v_ = vt_.t();
	Mat_<float> sparse = texture * v_.colRange(0, SPARSE_DIM);
	Mat_<float> mean_c(1, 9);
	mean_c.setTo(0);

	normalize(sparse, sparse, 0, 1, NORM_MINMAX);
	for (int i = 0; i < nLabels; ++i) {
		mean_c += color.row(i) / nLabels;
	}
	size_t sum_dims = sparse.cols + loc.cols + contra.cols + color.cols * 2;
	feats.create(nLabels, sum_dims);
	const int index[] = {
		0, 
		contra.cols, 
		contra.cols + loc.cols, 
		contra.cols + loc.cols + color.cols,  
		contra.cols + loc.cols + color.cols + sparse.cols, 
		contra.cols + loc.cols + color.cols + sparse.cols + mean_c.cols,
	};

	int t_ = 0;
	contra.copyTo(feats.colRange(index[t_], index[t_ + 1])); t_++; 
	   loc.copyTo(feats.colRange(index[t_], index[t_ + 1])); t_++;
	 color.copyTo(feats.colRange(index[t_], index[t_ + 1])); t_++;
	sparse.copyTo(feats.colRange(index[t_], index[t_ + 1])); t_++;
	for (int i = 0; i < nLabels; ++i) {
		mean_c.copyTo(feats.row(i).colRange(index[t_], index[t_+1]));
	}
	labels.clear();
	labels.resize(nLabels,0);
	if (gt_.empty())
		return ;

	Mat gt;
	if (gt_.channels() == 3) {
		cvtColor(gt_, gt, COLOR_RGB2GRAY);
	} 
	else {
		gt_.copyTo(gt);
	}
	if (gt.depth() == CV_8U) {
		gt.convertTo(gt, CV_32F);
		gt /= 255.0;
	}
	CV_Assert(gt.size() == rgb.size());
	//resize(gt, gt, Size(img_.cols, img_.rows), 0, 0, INTER_LINEAR);
	
	for (int i = 0; i < gt.rows; ++i) {
		for (int j = 0; j < gt.cols; ++j) {
			int id = seg_img(i, j);
			float num = (float)graph_.region_size(id);
			num = num > 0 ? num : 1;
			labels[id] += gt.at<float>(i, j) / num;
		}
	}

	// double min_, max_;
	// cv::minMaxLoc(labels, &min_, &max_);	
	// CV_Assert(min_ >= 0 && max_ <= 1);

}

void spStat::distFeatures(const Mat & img, const Mat & gt_, Mat_<int>& lbls, Mat_<float> &feats1,
	Mat_<float> &feats2,  vector<float> &dists) {
	
	Mat_<float> features;
	Mat_<int> labels;
	Mat_<int> seg_img;
	vector<float> lbl;
	extractFeatures(img, gt_, seg_img, features, lbl);
	labels.create(lbl.size(),2);
	labels.setTo(0);
	Mat mask1 = (Mat(lbl.size(), 1, CV_32F, lbl.data()) < 0.5);
	Mat mask2 = (Mat(lbl.size(), 1, CV_32F, lbl.data()) >= 0.5);
	(labels.col(0)).setTo(1, mask1);
	(labels.col(1)).setTo(1, mask2);

	float scale1 = 0.5;
	float scale2 = 0.9;
	Mat_<float> kernel;
	graph_.gen_Kernel(labels, kernel);
	int n = labels.rows;
	default_random_engine e((int)time(NULL));
	uniform_real_distribution<double> distribution(0, 1);
	auto dice = bind(distribution, e);

	int count = 0; 
	for (int i = 0; i < n; ++i) {
		if (labels(i, 0) == 1 && dice() > scale1)
			continue;
		if (labels(i, 1) == 1 && dice() > scale2)
			continue;
		set<int> surroundings;
		set<int> neighbours = graph_.region_neighbors(i);
		//surroundings.insert(neighbours.begin(), neighbours.end());
		for (set<int>::iterator it = neighbours.begin(); it != neighbours.end(); ++it) {
      		set<int> neighs = graph_.region_neighbors(*it);
			surroundings.insert(neighs.begin(), neighs.end());
      		surroundings.insert(*it);
		}
		
		//surroundings.insert(i);
		for (set<int>::iterator it = surroundings.begin(); it != surroundings.end(); ++it) {
			if (*it <= i) {
        		continue;
			}
			feats1.push_back(features.row(i));
			feats2.push_back(features.row(*it));
			dists.push_back(kernel(i, *it));
			Mat_<int> lbl_pair(1,2);
			lbl_pair(0,0) = labels(i, 1);
			lbl_pair(0,1) = labels(*it, 1);
			lbls.push_back(lbl_pair);
			count++;
		}
	}

	CV_Assert(feats1.rows == count);
	CV_Assert(feats2.rows == count);
	CV_Assert(dists.size() == count);
	clear();
}

vector<float> spStat::nodeContrast(const spGraph &graph, int flag) const {
	vector<float> contrasts(graph.size());
//#pragma omp parallel for
	for (int i = 0; i < graph.size(); ++i) {
		contrasts[i] = 0;
		float sum = 0.0;
		for (int j = 0; j < graph.size(); ++j) {
			Vec3f diff = (
				(flag == CONTRAST::RGB_CTR) ? graph.mean_color(i) - graph.mean_color(j) :
				( (flag == CONTRAST::LAB_CTR) ? graph.mean_lab(i) - graph.mean_lab(j) :
					graph.mean_hsv(i) - graph.mean_hsv(j) )
				);
			float color_dist = diff.dot(diff);
			float sp_dist = graph.eucliDist(i, j);
			contrasts[i] += exp(coeff_exp * sp_dist) * color_dist;
			sum += exp(coeff_exp * sp_dist);
		}
		//contrasts[i] /= sum;
	}
	normalize(contrasts, contrasts, 0, 1, NORM_MINMAX);
	return contrasts;
}

vector<float> spStat::nodeVariance(const spGraph &graph, int flag) const {
	vector<float> variances(graph.size());
	switch (flag) {
	case VARIANCE::HSV_VAR:
		for (int i = 0; i < variances.size(); ++i) {
			Vec3f diff = graph.mean_hsv(i) - graph.mean_hsv_g();
			variances[i] = (diff.dot(diff));
		}
		break;
	case VARIANCE::RGB_VAR:
		for (int i = 0; i < variances.size(); ++i) {
			Vec3f diff = graph.mean_color(i) - graph.mean_color_g();
			variances[i] = (diff.dot(diff));
		}
		break;
	case VARIANCE::LAB_VAR:
		for (int i = 0; i < variances.size(); ++i) {
			Vec3f diff = graph.mean_lab(i) - graph.mean_lab_g();
			variances[i] = (diff.dot(diff));
		}
		break;
	case VARIANCE::SPATIAL_VAR:
		for (int i = 0; i < variances.size(); ++i) {
			Vec2f diff = graph.mean_position(i) - graph.mean_position_g();
			variances[i] = (diff.dot(diff));
		}
		break;
	}
	
	normalize(variances, variances, 0, 1, NORM_MINMAX);
	return variances;
}

Mat spStat::testing(const Mat & img)
{
	Mat img_ = img.clone();
	float scale = maxBoundary / std::max(img_.rows, img_.cols);
	resize(img_, img_, Size(img_.cols * scale, img_.rows * scale), 0, 0, INTER_LINEAR);
	Mat temp = img_.clone();
	bilateralFilter(temp, img_, 3, 25, 50);
	
	Mat_<Vec3f> rgb;
	Mat sal_map1, sal_map;
	img.convertTo(rgb, CV_32FC3, 1 / 255.);

	spectralResidual(rgb, sal_map1, 256, 256);
	normalize(sal_map1, sal_map1, 0, 1, NORM_MINMAX);
	sal_map1.convertTo(sal_map1, CV_8UC3, 255);

	Mat color = rgb.clone();
	color.convertTo(color, CV_8UC3, 255);
	imshow("rgb", color);
	imshow(name_ + "1", sal_map1);
	waitKey(0);
	return Mat();
}

void spStat::gen_labels(vector<float>& labels, const Mat_<float> &res, const Mat_<int> &seg_img)
{
	CV_Assert(res.size() == seg_img.size());
	double max_;
	minMaxLoc(seg_img, NULL, &max_);
	labels.resize((int)(max_ + 1));
	//fill(labels.begin(), labels.end(), -1);
	for (int i = 0; i < res.rows; ++i) {
		for (int j = 0; j < res.cols; ++j) {
			int id = seg_img(i, j);
			labels[id] = res(i, j);
		}
	}
}

void spStat::gen_map(Mat &res, const vector<float>& labels, const Mat_<int> &seg_img)
{
	res.create(seg_img.size(), CV_32F);
	for (int i = 0; i < seg_img.rows; ++i) {
		for (int j = 0; j < seg_img.cols; ++j) {
			int id = seg_img(i, j);
			res.at<float>(i, j) = labels[id];
		}
	}
}

pair<Mat_<float>, vector<float>> spStat::interpolating(const Mat_<float>& features, const vector<float>& labels, 
		int size) {
	Mat_<float> spatial = features.colRange(27, 29).clone();
	spatial = spatial * size;
	Mat_<float> output(size * size * 4, features.cols);
    output.setTo(0.0);
	vector<float> C(size * size * 4);
	vector<float> nlabels(size * size * 4);
	float offset[] = {0.25, 0.75};
    int dim = 2;    

	for (int j = 0; j < dim; ++j) {
		for (int i = 0; i < dim; ++i) {
            for (int k = 0; k < features.rows; ++k) {
                for (int y = static_cast<int>(spatial(k, 1)); y < size && y <= static_cast<int>(spatial(k, 1))+1; ++y) {
                    for (int x = static_cast<int>(spatial(k, 0)); x < size && x <= static_cast<int>(spatial(k, 0))+1; ++x) {
		        		float dw = std::fabs(spatial(k,0) - x - offset[i]);
	        			float dh = std::fabs(spatial(k,1) - y - offset[j]);
	        			int id = (y * dim + j) * size * dim + x * dim + i;
	        			output.row(id) += dw * dh * features.row(k);
	        			nlabels[id] += dw * dh * labels[k];
	        			C[id] += dw * dh;
                    }
                }
			}
		}
	}
	for (int i = 0; i < output.rows; ++i) {
		output.row(i) /= C[i] + 1e-20;
		nlabels[i] /= C[i] + 1e-20;
	}

	return pair<Mat_<float>, vector<float>>(output, nlabels);
}

void spStat::backproject(const Mat_<float>& features, const vector<float>& nlabels, vector<float>& labels, int size) {
	;// NOT_INPLEMENT
}
