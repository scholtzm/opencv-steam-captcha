#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define HISTSIZE 256

Mat createHistogram(Mat& source) {
    Mat histogram;

    int histSize = HISTSIZE;
    float range[] = { 0, 256 };
    const float *ranges[] = { range };
    calcHist(&source, 1, 0, Mat(), histogram, 1, &histSize, ranges, true, false);
    
    return histogram;
}

Mat drawHistogram(Mat& histogram, int marker) {
    int histHeight = 512;
    int histWidth = 512;
    int widthBucket = cvRound((double) histWidth/HISTSIZE);
    
    Mat histogramImage = Mat(histHeight, histWidth, CV_32FC3, Scalar(0));
    normalize(histogram, histogram, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());
    
    for(int i = 1; i < HISTSIZE; i++)
    {
        line(histogramImage, Point(widthBucket * (i - 1), histHeight - cvRound(histogram.at<float>(i - 1))) ,
             Point(widthBucket * i, histHeight - cvRound(histogram.at<float>(i))),
             Scalar(255, 255, 255), 1);
    }
    
    if(marker >= 0)
        line(histogramImage, Point(widthBucket * marker, 0), Point(widthBucket * marker, histHeight - 1), Scalar(0, 255, 0), 1);
    
    return histogramImage;
}

int getIdealThreshold(Mat& histogram) {
    const int MAGIC_PEAK = 200;
    const int MAGIC_PERCENTAGE = 4;
    
    int lastPeak = 0;
    
    for(int i = 0; i < HISTSIZE; i++) {
        if(cvRound(histogram.at<float>(i)) > MAGIC_PEAK) {
            lastPeak = i;
        }
    }
    
    return lastPeak + (int)((255 - lastPeak) / 10 * MAGIC_PERCENTAGE);
}