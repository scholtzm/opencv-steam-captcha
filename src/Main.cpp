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

int dynamicThreshold = 100;
string windowName = "Steam Captcha Breaker";
Mat sourceImage, finalImage, histImage;
Mat hist;

bool compareContours(vector<Point> a, vector<Point> b);
void applyThreshold(int, void*);
void createHistogram(int threshold);

bool compareContours(vector<Point> a, vector<Point> b) {
	return contourArea(a) > contourArea(b);
}

void createHistogram(Mat& src, int threshold) {
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
    
    line(histImage, Point(widthBucket * threshold, 0), Point(widthBucket * threshold, histHeight - 1), Scalar(0, 255, 0), 1);
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

void applyThreshold(int, void*) {
    threshold(sourceImage, finalImage, dynamicThreshold, 255, THRESH_BINARY);
    
    createHistogram(sourceImage, dynamicThreshold);
    
    imshow("Histogram", histImage);
    imshow(windowName, finalImage);
}

Mat applyCanny(Mat& src, int threshold) {
    Mat edges, tmp;
    
    Canny(src, edges, threshold, threshold * 3, 3);
    
    src.copyTo(tmp, edges);
    tmp.copyTo(src);
    
    edges.release();
    tmp.release();
    
    return src;
}

int main(int argc, char** argv) {
	const int BYTES_PER_PIXEL = 1;
	const int RESIZE_FACTOR = 2;
	const string DATA_PATH = "/Users/Mike/Documents/Eclipse Workspace/SCB3/data/";

	int total = 0;
	int success = 0;

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
		cout << total << ".) file: " << fileName << ", code: " << captchaCode;

		// Load our base image
		sourceImage = imread(fullPath, CV_LOAD_IMAGE_GRAYSCALE);
        
		// Is it loaded?
		if(!sourceImage.data)
			return -1;
        
        resize(sourceImage, sourceImage, Size(sourceImage.cols * RESIZE_FACTOR, sourceImage.rows * RESIZE_FACTOR));
        // Normalize our image
        normalize(finalImage, finalImage, 0, 255, NORM_MINMAX, CV_8U);

		// Define our final image
		finalImage = sourceImage.clone();
        
        // Apply adaptive threshold
        adaptiveThreshold(finalImage, finalImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 1);
        
        Mat tmp;
        sourceImage.copyTo(tmp, finalImage);
        tmp.copyTo(finalImage);
        
        normalize(finalImage, finalImage, 0, 255, NORM_MINMAX, CV_8U);
        
		// Let's apply image reconstruction
//		finalImage -= 30;
//		ImageReconstruct<unsigned char>(finalImage, sourceImage);
        
//        imshow("Test", (sourceImage - finalImage) * 1);

		// Let's calculate histogram for our image
        createHistogram(finalImage, dynamicThreshold);

	    int thresholdValueLow = 0;
	    for(int i = 0; i < 256; i++) {
	    	if(cvRound(hist.at<float>(i)) > 10) {
	    		thresholdValueLow = i;
#if SMOKETEST == 0
                cout << "LOW: " << thresholdValueLow << endl;
#endif
	    	}
	    }

	    // Calculate final threshold value
	    int thresholdValue = thresholdValueLow + (int)((255 - thresholdValueLow) / 10 * 4);
        
#if SMOKETEST == 0
        cout << "THRESH: " << thresholdValue << endl;
#endif
        
		// Apply threshold
		threshold(finalImage, finalImage, thresholdValue, 255, THRESH_BINARY);
        
        // Segments
        int* segH = horizontalSegments(finalImage);
        int* segV = verticalSegments(finalImage);
        
        Mat segHImage = drawHorizontalSegments(segH, finalImage.rows, finalImage.cols);
        Mat segVImage = drawVerticalSegments(segV, finalImage.rows, finalImage.cols);
        
        // Let's draw the rectangles
        vector<pair<int, int> > verticalPairs = createSegmentPairs(segV, finalImage.rows);
        vector<pair<int, int> > horizontalPairs = createSegmentPairs(segH, finalImage.cols);
        
        for(vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
            for(vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
                rectangle(finalImage, Point(itH->first, itV->first), Point(itH->second, itV->second), Scalar(255));
            }
        }

//		// Let's find contours
//		vector<vector<Point> > contours;
//		Mat contourImage = finalImage.clone();
//		findContours(contourImage, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
//
//		// Sort them by size
//		sort(contours.begin(), contours.end(), compareContours);
//
//		// Hide noise by filling it with black color
//        int contourCount = MIN((int) contours.size(), 6);
//            if(contours.size() > contourCount) {
//                for(int contourIndex = contourCount; contourIndex < contours.size(); contourIndex++) {
//                    drawContours(finalImage, contours, contourIndex, Scalar(0), CV_FILLED);
//                }
//        }

		// Initiate tesseract
		tesseract::TessBaseAPI tess;
		tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
		tess.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ@%&");
		tess.SetImage((uchar *) finalImage.data, finalImage.cols, finalImage.rows, BYTES_PER_PIXEL, BYTES_PER_PIXEL * finalImage.cols);

		// Retrieve the result and remove all spaces
		string result(tess.GetUTF8Text());
		result.erase(remove_if(result.begin(), result.end(), ::isspace), result.end());

		cout << ", result: " << result;

		if(captchaCode == result) {
			cout << " - TRUE" << endl;
			success++;
		} else {
			cout << " - FALSE" << endl;
		}

		// Resize back
//		resize(sourceImage, sourceImage, Size(sourceImage.cols / RESIZE_FACTOR, sourceImage.rows / RESIZE_FACTOR));
//		resize(finalImage, finalImage, Size(finalImage.cols / RESIZE_FACTOR, finalImage.rows / RESIZE_FACTOR));

#if SMOKETEST == 0
//        namedWindow(windowName);
//        finalImage.copyTo(sourceImage);
//        
//        createTrackbar("Value", windowName, &dynamicThreshold, 255, applyThreshold);
//        
//        applyThreshold(0, 0);
//        
//        while(true)
//        {
//            int c;
//            c = waitKey();
//            if((char)c == 27) break;
//        }
        
        imshow("Final image", finalImage);
        imshow("HSeg", segHImage);
        imshow("VSeg", segVImage);
        waitKey();
#endif

		sourceImage.release();
		finalImage.release();
//		contourImage.release();
//		contours.clear();

#if SMOKETEST == 0
		if(total == 10) break;
#endif
	}

	cout << "Success rate: " << ((double)success/(double)total)*100 << "% (" << success << "/" << total << ")." << endl;

	return 0;
}
