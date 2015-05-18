#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void getSimpleDescriptors(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize) {
    Mat image;
    
    // Get dimensions first
    image = imread(folder + posLetter + "/0.png", CV_LOAD_IMAGE_GRAYSCALE);
    
    trainingData = Mat(sampleSize * 2, image.cols * image.rows, CV_32FC1);
    classLabels = Mat(sampleSize * 2, 1, CV_32FC1);
    
    image.release();
    
    for(int i = 0; i < sampleSize; i++) {
        image = imread(folder + posLetter + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        if (!image.data)
            break;
        
        for(int j = 0; j < image.rows; j++)
            for(int k = 0; k < image.cols; k++)
                trainingData.at<float>(i * 2, j * image.cols + k) = ((float) image.at<unsigned char>(j, k)) / 255.0;
        
        classLabels.at<float>(i * 2, 0) = 1.0;
        
        image.release();
        
        image = imread(folder + negLetter + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        if (!image.data)
            break;
        
        for(int j = 0; j < image.rows; j++)
            for(int k = 0; k < image.cols; k++)
                trainingData.at<float>(i * 2 + 1, j * image.cols + k) = ((float) image.at<unsigned char>(j, k)) / 255.0;
        
        classLabels.at<float>(i * 2 + 1, 0) = -1.0;
        
        image.release();
    }
}

void getHOGDescriptors(Mat& trainingData, Mat& classLabels, string folder, string posLetter, string negLetter, int sampleSize) {
    Mat image;
    vector<float> features;
    
    trainingData = Mat(sampleSize * 2, 540, CV_32FC1);
    classLabels = Mat(sampleSize * 2, 1, CV_32FC1);
    
    for(int i = 0; i < sampleSize; i++) {
        HOGDescriptor hog;
        hog.winSize = Size(32, 48);
        
        image = imread(folder + posLetter + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        if (!image.data)
            break;
        hog.compute(image, features, Size(8, 8), Size(0, 0));
        
        for(int j = 0; j < features.size(); j++)
            trainingData.at<float>(i * 2, j) = features[j];
        
        classLabels.at<float>(i * 2, 0) = 1.0;
        
        image = imread(folder + negLetter + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        if (!image.data)
            break;
        hog.compute(image, features, Size(8, 8), Size(0, 0));
        
        for(int j = 0; j < features.size(); j++)
            trainingData.at<float>(i * 2 + 1, j) = features[j];
        
        classLabels.at<float>(i * 2 + 1, 0) = -1.0;
    }
}