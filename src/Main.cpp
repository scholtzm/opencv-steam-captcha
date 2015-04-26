#include <algorithm>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "imagereconstruct.hpp"

#define SMOKETEST 0

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

Mat sourceImage, finalImage, histImage, tmp;
Mat hist;

void createHistogram(Mat& src) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float *ranges[] = { range };
    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
    
    // Let's draw our histogram
    int histHeight = 512;
    int histWidth = 512;
    int widthBucket = cvRound((double) histWidth/histSize);
    
    histImage = Mat(histHeight, histWidth, CV_32FC3, Scalar(0));
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    
    for(int i = 1; i < histSize; i++)
    {
        line(histImage, Point(widthBucket * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))) ,
             Point(widthBucket * i, histHeight - cvRound(hist.at<float>(i))),
             Scalar(255, 255, 255), 1);
    }
}

int* horizontalSegments(Mat& src) {
    int* seg = (int*) calloc(src.cols, sizeof(int));
    
    for(int i = 0; i < src.cols; i++) {
        for(int k = 0; k < src.rows; k++) {
            uchar pixel = src.data[src.cols * k + i];
            
            if(pixel)
                seg[i]++;
        }
    }
    
    return seg;
}

Mat drawHorizontalSegments(int* seg, int rows, int cols) {
    Mat segImage = Mat(rows, cols, CV_8U, Scalar(0));
    
    for(int i = 0; i < cols; i++) {
        if(seg[i] > 0)
            line(segImage, Point(i, rows - 1), Point(i, rows - 1 - seg[i]), Scalar(255), 1);
    }
    
    return segImage;
}

int* verticalSegments(Mat& src) {
    int* seg = (int*) calloc(src.rows, sizeof(int));
    
    for(int i = 0; i < src.rows; i++) {
        for(int k = 0; k < src.cols; k++) {
            uchar pixel = src.data[src.cols * i + k];
            
            if(pixel)
                seg[i]++;
        }
    }
    
    return seg;
}

Mat drawVerticalSegments(int* seg, int rows, int cols) {
    Mat segImage = Mat(rows, cols, CV_8U, Scalar(0));
    
    for(int i = 0; i < rows; i++) {
        if(seg[i] > 0)
            line(segImage, Point(0, i), Point(seg[i], i), Scalar(255), 1);
    }
    
    return segImage;
}

vector<pair<int, int> > createSegmentPairs(int* seg, int segSize) {
    int top = 0, bottom = 0;
    bool in = false;
    
    vector<pair<int, int> > pairs;
    
    for(int i = 0; i < segSize; i++) {
        if(seg[i])
            in = true;
        else
            in = false;
        
        if(in) {
            if(!top) {
                top = i;
                bottom = i;
            } else {
                bottom = i;
            }
            
            // corner case
            if(i == segSize - 1) {
                pairs.push_back(make_pair(top, bottom));
                top = bottom = 0;
            }
        } else {
            if(top && bottom) {
                pairs.push_back(make_pair(top, bottom));
                top = bottom = 0;
            }
        }
    }
    
    
    return pairs;
}

vector<pair<int, int> > filterVerticalPairs(vector<pair<int, int> > verticalPairs) {
    vector<pair<int, int> > new_pairs;
    
    int biggest = 0, index = 0;
    
    for(int i = 0; i < verticalPairs.size(); i++) {
        int current = verticalPairs[i].second - verticalPairs[i].first;
        
        if(current > biggest) {
            index = i;
        }
    }
    
    new_pairs.push_back(make_pair(verticalPairs[index].first, verticalPairs[index].second));
    
    return new_pairs;
}

vector<pair<int, int> > filterHorizontalPairs(vector<pair<int, int> > horizontalPairs, int segSize) {
    int MAGIC_DIFF = 10;
    int MAGIC_SIZE = 10;
    
    vector<pair<int, int> > new_pairs;
    
    int* seg = (int*) calloc(segSize, sizeof(int));
    
    for(int i = 0; i < horizontalPairs.size(); i++) {
        int size = horizontalPairs[i].second - horizontalPairs[i].first;
        
        // Too small
        if(size < MAGIC_SIZE) {
            // Check right side
            if(i < horizontalPairs.size() - 1 && horizontalPairs.size() > 1) {
                if(horizontalPairs[i + 1].first - horizontalPairs[i].second < MAGIC_DIFF) {
                    for(int k = horizontalPairs[i].first; k <= horizontalPairs[i + 1].second; k++)
                        seg[k] = 1;
                }
                // Check left side
            } else if(i > 0 && horizontalPairs.size() > 1) {
                if(horizontalPairs[i].first - horizontalPairs[i - 1].second < MAGIC_DIFF) {
                    for(int k = horizontalPairs[i - 1].first; k <= horizontalPairs[i].second; k++)
                        seg[k] = 1;
                }
            }
            // This one is large enough
        } else {
            for(int k = horizontalPairs[i].first; k <= horizontalPairs[i].second; k++)
                seg[k] = 1;
        }
    }
    
    return createSegmentPairs(seg, segSize);
}

int main(int argc, char** argv) {
    const int RESIZE_FACTOR = 2;
    const string DATA_PATH = "/Users/Mike/Documents/Eclipse Workspace/SCB3/data/";
    
    int total = 0;
    
    fs::path folder(DATA_PATH);
    
    if(!exists(folder))
        return -1;
    
    fs::directory_iterator endItr;
    for(fs::directory_iterator itr(folder); itr != endItr; itr++) {
        string fullPath = itr->path().string();
        string fileName = itr->path().filename().string();
        
        // Skip all dot files
        if(fileName[0] == '.')
            continue;
        
        // Retrieve captcha string
        string captchaCode = boost::replace_all_copy(fileName, ".png", "");
        boost::replace_all(captchaCode, "at", "@");
        boost::replace_all(captchaCode, "pct", "%");
        boost::replace_all(captchaCode, "and", "&");
        
        total++;
        cout << total << ".) file: " << fileName << ", code: " << captchaCode << endl;
        
        // Load our base image
        sourceImage = imread(fullPath, CV_LOAD_IMAGE_GRAYSCALE);
        
        // Is it loaded?
        if(!sourceImage.data)
            return -1;
        
        // Resize the image 2x
        resize(sourceImage, sourceImage, Size(sourceImage.cols * RESIZE_FACTOR, sourceImage.rows * RESIZE_FACTOR));
        
        // Define our final image
        finalImage = sourceImage.clone();
        
        // Apply adaptive threshold
        adaptiveThreshold(finalImage, finalImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 1);
        
        // Use the thresholded image as a mask
        sourceImage.copyTo(tmp, finalImage);
        tmp.copyTo(finalImage);
        
        // Normalize new image
        normalize(finalImage, finalImage, 0, 255, NORM_MINMAX, CV_8U);
        
        // Let's calculate histogram for our image
        createHistogram(finalImage);
        
        // Get lower bound of our threshold value
        int thresholdValueLow = 0;
        for(int i = 0; i < 256; i++) {
            if(cvRound(hist.at<float>(i)) > 4) {
                thresholdValueLow = i;
            }
        }
        
        // Calculate final threshold value
        int thresholdValue = thresholdValueLow + (int)((255 - thresholdValueLow) / 10 * 4);
        
        // Apply threshold
        threshold(finalImage, finalImage, thresholdValue, 255, THRESH_BINARY);
        
        // Apply dilation and erosion
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        dilate(finalImage, finalImage, element);
        erode(finalImage, finalImage, element);
        
        // Segments
        int* segH = horizontalSegments(finalImage);
        int* segV = verticalSegments(finalImage);
        
        Mat segHImage = drawHorizontalSegments(segH, finalImage.rows, finalImage.cols);
        Mat segVImage = drawVerticalSegments(segV, finalImage.rows, finalImage.cols);
        
        // Let's draw the rectangles
        vector<pair<int, int> > verticalPairs = filterVerticalPairs(createSegmentPairs(segV, finalImage.rows));
        vector<pair<int, int> > horizontalPairs = filterHorizontalPairs(createSegmentPairs(segH, finalImage.cols), finalImage.cols);
        
        for(vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
            for(vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
                rectangle(finalImage, Point(itH->first, itV->first), Point(itH->second, itV->second), Scalar(255));
            }
        }
        
        imshow("Final image", finalImage);
        //        imshow("HSeg", segHImage);
        //        imshow("VSeg", segVImage);
        imshow("Histogram", histImage);
        waitKey();
        
        sourceImage.release();
        finalImage.release();
        tmp.release();
#if SMOKETEST == 0
        if(total == 10) break;
#endif
    }
    
    return 0;
}
