#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat createHistogram(Mat& source);
Mat drawHistogram(Mat& histogram, int marker);
int getIdealThreshold(Mat& histogram);

#endif