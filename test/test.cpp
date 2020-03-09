#include <iostream>
#include "definition.h"
#include "spStat.h"
#include <string>
#include "slic.h"
#include "crfinference.h"
using namespace std;
using namespace cv;

void test_slic(string img_name = "../fig/1.png") {
    Mat img = imread(img_name);
    resize(img, img, Size(400,400));
    Slic slic;
    Mat lab;
    cvtColor(img, lab, COLOR_RGB2Lab);
    Mat seg_img = slic.generate_superpixels(lab, -1, 40, 400);
    //seg_img.convertTo(seg_img, CV_8UC1);
    double max, min;
    minMaxLoc(seg_img, &min, &max);
    vector<int> counts((int)(max + 1));
    vector<Vec3f> colors((int)(max + 1));
    Mat simg = Mat::zeros(img.size(), CV_8UC3);
    //slic.colour_with_cluster_means(simg);
    for (int i = 0; i < seg_img.rows; ++i) {
        for (int j = 0; j < seg_img.cols; ++j) {
            int id = seg_img.at<int>(i, j);
            counts[id]++;
            for (int k = 0; k < 3; ++k)
                colors[id].val[k] += img.at<Vec3b>(i,j).val[k];
        }
    }
    for (int i = 0; i < seg_img.rows; ++i) {
        for (int j = 0; j < seg_img.cols; ++j) {
            int id = seg_img.at<int>(i, j);
            for (int k = 0; k < 3; ++k)
                simg.at<uchar>(i, j * 3 + k) = colors[id].val[k] / counts[id];
        }
    }
    imshow("img", img);
    imshow("slic", simg);
    waitKey(0);
    return ;
}

int main(int argc, char **argv) {
    // vector<string> names = {"1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"};
    // for (auto name : names)
    //     test_slic("../fig/" + name);
    // return 0;
    if (argc <= 1) {
        return -1;
    }
    cout << "image is reading" << endl;
    string img_name = string(argv[1]);
    Mat img = imread(img_name);
    Mat gt = imread(img_name.substr(0, img_name.size() - 3) + "png");
    {
        gt.convertTo(gt, CV_32FC3);
        gt /= 255.0;
        cvtColor(gt, gt, COLOR_RGB2GRAY);
        cout << (gt.depth() == CV_32F ? "cv32f" : "cv8u") << endl;
        cout << "channels: " << gt.channels() << endl;
    }
    resize(img, img, Size(400,400));
    resize(gt, gt, Size(400,400));
    cout << "image is read" << endl;
    spStat stat;
    Mat_<float> feats;
    Mat_<int> seg;
    vector<float> labels;
    stat.extractFeatures(img, gt, seg, feats, labels);
    cout << "features' size: " << feats.size() << endl;
    cout << "seg' size: " << seg.size() << endl;
    double max_val, min_val;
    minMaxLoc(seg, &min_val, &max_val);
    cout << "max, min: " << max_val << ", " << min_val << endl;
    
    pair<Mat_<float>, vector<float>> pairs = stat.interpolating(feats, labels);
    cout << pairs.first.size() << endl;
    Mat map(40, 40, CV_8U);
    for (int i = 0; i < 40; ++i) {
        for (int j = 0; j < 40; ++j) {
            map.at<uchar>(i, j) = pairs.second[i * 40 + j] * 255;
        }
    }
    Mat mask(img.size(), CV_8U);
    Mat Kernel = Mat::ones(max_val + 1, max_val + 1, CV_32F) - 0.5;
    RNG r;
    r.fill(mask, RNG::UNIFORM, 0, 256);
    Mat res = CrfInference(img, mask, Kernel, seg);
    cout << res.size() << endl;

    resize(map, map, Size(256,256));
    imshow("map", map);
    waitKey(0);
    return 0;
}
