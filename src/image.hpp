#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Creates histogram from our 8 bit image.
 */
Mat createHistogram(Mat& source);

/**
 * Draws histogram to a 512 x 512 image.
 */
Mat drawHistogram(Mat& histogram, int marker);

/**
 * Attempts to calculate ideal threshold from histogram data.
 */
int getIdealThreshold(Mat& histogram);

#endif