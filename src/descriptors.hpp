#ifndef _DESCRIPTORS_H_
#define _DESCRIPTORS_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void getSimpleDescriptor(Mat& image, Mat& descriptor);
void getHOGDescriptor(Mat& image, Mat& descriptor);

void getSimpleDescriptors(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize);
void getHOGDescriptors(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize);

#endif