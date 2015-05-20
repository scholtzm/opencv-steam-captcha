#ifndef _DESCRIPTORS_H_
#define _DESCRIPTORS_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Functions to generate descriptor for a single image.
 */
void getSimpleDescriptor(Mat& image, Mat& descriptor);
void getHOGDescriptor(Mat& image, Mat& descriptor);

/**
 * Functions to generate training data and appropriate labels.
 */
void getSimpleTrainingData(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize);
void getHOGTrainingData(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize);

#endif