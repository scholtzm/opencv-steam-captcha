#ifndef _CLASSIFY_H_
#define _CLASSIFY_H_

#include <opencv2/opencv.hpp>

/**
 * Very simple classification function that uses SVM.
 */
float classify(CvSVM& SVM, Mat& image);

#endif