#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat createHistogram(Mat& source, Mat& histogramImage) {
    Mat histogram;

    int histSize = 256;
    float range[] = { 0, 256 };
    const float *ranges[] = { range };
    calcHist(&source, 1, 0, Mat(), histogram, 1, &histSize, ranges, true, false);
    
    // Let's draw our histogram
    int histHeight = 512;
    int histWidth = 512;
    int widthBucket = cvRound((double) histWidth/histSize);
    
    histogramImage = Mat(histHeight, histWidth, CV_32FC3, Scalar(0));
    normalize(histogram, histogram, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());
    
    for(int i = 1; i < histSize; i++)
    {
        line(histogramImage, Point(widthBucket * (i - 1), histHeight - cvRound(histogram.at<float>(i - 1))) ,
             Point(widthBucket * i, histHeight - cvRound(histogram.at<float>(i))),
             Scalar(255, 255, 255), 1);
    }
    
    return histogram;
}