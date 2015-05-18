#ifndef _DESCRIPTORS_H_
#define _DESCRIPTORS_H_

#include <opencv2/opencv.hpp>

void getSimpleDescriptors(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize);
void getHOGDescriptors(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize);

#endif