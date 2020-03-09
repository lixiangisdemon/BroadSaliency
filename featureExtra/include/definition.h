#ifndef __DEFINITION_H__ 
#define __DEFINITION_H__ 

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
extern "C" {
#undef min
#undef max
}
using namespace std;
using namespace cv;

typedef Vec<float, 9> Vec9f;

inline std::ostream & operator << (std::ostream &out, std::pair<int, int> x) {
	std::cout << "(" << x.first << "," << x.second << ")";
	return out;
}

template <class T>
inline std::ostream & operator << (std::ostream &out, std::vector<T> &vec) {
	if (vec.empty()) {
		std::cout << "[]" << std::endl;
		return out;
	}
	int size_ = vec.size() ;
	std::cout << "[";
	for (int i = 0; i < size_; i++) {
		if (i == 7 && i + 7 < size_) {
			std::cout << "... ";
			i = size_ - 7;	
		}
		std::cout << vec[i];
		if (i < size_ -1)
			std::cout << ",";
	}
	std::cout << "]";
	return out;
}

inline void normLab(Mat_<Vec3f> &lab) {
	for (Mat_<Vec3f>::iterator it = lab.begin(); it != lab.end(); it++) {
		(*it)[0] /= 100;
		(*it)[1] = ((*it)[1] + 127) / 255;
		(*it)[2] = ((*it)[2] + 127) / 255;
	}
}

inline void normHSV(Mat_<Vec3f> &hsv) {
	for (Mat_<Vec3f>::iterator it = hsv.begin(); it != hsv.end(); it++) {
		(*it)[0] /= 360;
		(*it)[1] /= 100;
		(*it)[2] /= 100;
	}
}

inline void normRGB(Mat_<Vec3f> &rgb) {
	rgb /= 255.;
}

inline void adjustHSV(Mat_<Vec3f> &hsv) {
	for (Mat_<Vec3f>::iterator it = hsv.begin(); it != hsv.end(); it++) {
		(*it)[1] *= 100;
		(*it)[2] *= 100;
	}
}

inline string get_last_name(const string str) {
	int len = str.size();
	int name_ptr = 0;
	int name_len = 0;
	for (int i = 0; i < len; i++) {
		if (str[i] == '/') {
			name_ptr = i + 1;
			name_len = 0;
		}
		name_len++;
	}
	return str.substr(name_ptr, name_len);
}

template<typename T>
inline Mat_<T> read_Mats_(string str) {
	fstream fb(str, ios::in);
	Mat_<T> matrix;
	while (!fb.eof()) {
		Mat_<T> vec;
		stringstream ss;
		string str;
		getline(fb, str);
		ss << str;
		while (!ss.eof()) {
			T tmp;
			ss >> tmp;
			vec.push_back(tmp);
		}
		matrix.push_back(vec.t());
	}
	fb.close();
	return matrix;
}

template<typename T>
inline void write_Mat_(string str, const Mat_<T> &matrix) {
	fstream fb(str, ios::out);
	for (int k = 0; k < matrix.rows; k++) {
		for (int l = 0; l < matrix.cols; l++) {
			fb << matrix(k, l);
			if (l < matrix.cols - 1) {
				fb << " ";
			}
		}
		if (k < matrix.rows - 1) {
			fb << endl;
		}
		if (k % 5000 == 0 && k != 0) {
			cout << k << " rows have finished!" << endl;
		}
	}
}

template<typename T>
inline void S_write_Mat_(string str, const Mat_<T> &matrix, const vector<int> indexes) {
	if (indexes.empty()) {
    write_Mat_<T>(str, matrix);
		return ;
	}
	fstream fb(str, ios::out);
	for (int i = 0; i < indexes.size(); i++) {
	  int k = indexes[i];
		for (int l = 0; l < matrix.cols; l++) {
			fb << matrix(k, l);
			if (l < matrix.cols - 1) {
				fb << " ";
			}
		}
		if (i < indexes.size() - 1) {
			fb << endl;
		}
		if (i % 5000 == 0 && i != 0) {
			cout << i << " rows have finished!" << endl;
		}
	}
}

#endif