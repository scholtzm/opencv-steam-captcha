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

struct Square {
    int id; // position in the string
    int x;
    int y;
    int width;
    int height;
};

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
            biggest = current;
        }
    }
    
    new_pairs.push_back(make_pair(verticalPairs[index].first, verticalPairs[index].second));
    
    return new_pairs;
}

vector<pair<int, int> > filterHorizontalPairs(vector<pair<int, int> > horizontalPairs, int segSize) {
    const int MAGIC_DIFF = 10;
    const int MAGIC_SIZE = 12;
    
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
            }
            
            // Check left side
            if(i > 0 && horizontalPairs.size() > 1) {
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

vector<pair<int, int> > splitLarge(vector<pair<int, int> > horizontalSegments) {
    const int MAGIC_SIZE = 50;
    
    vector<pair<int, int> > new_pairs;
    
    for(int i = 0; i < horizontalSegments.size(); i++) {
        int size = horizontalSegments[i].second - horizontalSegments[i].first;
        
        if(size > MAGIC_SIZE) {
            int f1 = horizontalSegments[i].first;
            int s1 = (int)((horizontalSegments[i].second - horizontalSegments[i].first) / 2) + horizontalSegments[i].first;
            int f2 = (int)((horizontalSegments[i].second - horizontalSegments[i].first) / 2) + horizontalSegments[i].first + 2;
            int s2 = horizontalSegments[i].second;
            
            new_pairs.push_back(make_pair(f1, s1));
            new_pairs.push_back(make_pair(f2, s2));
        } else {
            new_pairs.push_back(horizontalSegments[i]);
        }
    }
    
    return new_pairs;
}

void drawSegmentRectangles(Mat& image, vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs) {
    for(vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
        for(vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
            rectangle(image, Point(itH->first, itV->first), Point(itH->second, itV->second), Scalar(255));
        }
    }
}

vector<Square> getSquares(vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs) {
    vector<Square> squares;
    int id = 0;
    
    for(vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
        for(vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
            squares.push_back({ id++, itH->first, itV->first, itH->second - itH->first, itV->second - itV->first });
        }
    }
    
    return squares;
}

// image must be a binary image
vector<Square> shrinkSquares(Mat& image, vector<Square> squares) {
    vector<Square> new_squares;
    
    for(int i = 0; i < squares.size(); i++) {
        int top = -1, bottom = 0, left = 9999, right = -1;
        Mat tmp2 = image(Rect(squares[i].x, squares[i].y, squares[i].width, squares[i].height));
        Mat tmp = tmp2.clone();
        
        for(int y = 0; y < tmp.rows; y++) {
            for(int x = 0; x < tmp.cols; x++) {
                int pixel = tmp.data[x + y * tmp.cols];
                
                if(pixel) {
                    tmp.data[x + y * tmp.cols] = 127;
                    if(top == -1) top = y; // store the lowest value
                    if(left > x) left = x;
                    bottom = y; // y will be always incremented, we just store the highest value
                    if(right < x) right = x;
                }
            }
        }
        
        top -= 1;
        left -= 1;
        bottom += 1;
        right += 1;
        
        new_squares.push_back({ squares[i].id, squares[i].x + left, squares[i].y + top, right - left, bottom - top });
        tmp.release();
        tmp2.release();
    }
    
    return new_squares;
}

bool compareSquaresBySize(Square a, Square b) {
    return a.width * a.height > b.width * b.height;
}

bool compareSquaresById(Square a, Square b) {
    return a.id < b.id;
}

vector<Square> takeSquares(vector<Square> squares, int number) {
    vector<Square> new_squares;
    
    sort(squares.begin(), squares.end(), compareSquaresBySize);
    int min = MIN(number, (int)squares.size());
    
    for(int i = 0; i < min; i++) {
        new_squares.push_back(squares[i]);
    }
    
    sort(new_squares.begin(), new_squares.end(), compareSquaresById);
    
    return new_squares;
}

void drawSquares(Mat& image, vector<Square> squares) {
    for(int i = 0; i < squares.size(); i++)
        rectangle(image, Point(squares[i].x, squares[i].y), Point(squares[i].x + squares[i].width, squares[i].y + squares[i].height), Scalar(255));
}

void saveSquares(Mat& image, vector<Square> squares, string output_folder, string code, map<string, int>& counter) {
    for(int i = 0; i < squares.size(); i++) {
        Mat tmp = image(Rect(squares[i].x, squares[i].y, squares[i].width, squares[i].height));
        Mat tmp2 = tmp.clone();
        
        string letter(1, code[i]);
        
        string sub = letter;
        boost::replace_all(sub, "@", "at");
        boost::replace_all(sub, "%", "pct");
        boost::replace_all(sub, "&", "and");
        
        string filename = output_folder + sub + "/" + to_string(counter[letter]) + ".png";
        counter[letter]++;
        
        resize(tmp2, tmp2, Size(50, 50));
        
        imwrite(filename, tmp2);
        
        tmp.release();
        tmp2.release();
    }
}

bool createFolderStructure(string output_folder) {
    std::vector<string> strings = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z", "at", "and", "pct"
    };
    
    fs::path folder(output_folder);
    if(!exists(folder) && !fs::create_directory(folder))
        return false;
    
    for(int i = 0; i < strings.size(); i++) {
        fs::path dir(output_folder + strings[i]);
        
        if(exists(dir))
            continue;
        
        if(!fs::create_directory(dir))
            return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    const int RESIZE_FACTOR = 2;
    const string DATA_PATH = "/Users/Mike/Documents/Eclipse Workspace/SCB3/data/";
    const string OUTPUT_PATH = "/Users/Mike/Documents/Eclipse Workspace/SCB3/output/";
    const string ALLOWED_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ@&%";
    
    int total = 0;
    
    // Initialize character counters
    map<string, int> counter;
    for(int i = 0; i < ALLOWED_CHARS.length(); i++) {
        string letter(1, ALLOWED_CHARS[i]);
        counter[letter] = 0;
    }
    
    // Check if data folder exists
    fs::path folder(DATA_PATH);
    if(!exists(folder))
        return -1;
    
    // Create output folder structure
    if(!createFolderStructure(OUTPUT_PATH))
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
        
        // Resize the image by resize factor
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
        
        // Morphological opening
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        dilate(finalImage, finalImage, element);
        erode(finalImage, finalImage, element);
        
        // Segments
        int* segH = horizontalSegments(finalImage);
        int* segV = verticalSegments(finalImage);
        
        Mat segHImage = drawHorizontalSegments(segH, finalImage.rows, finalImage.cols);
        Mat segVImage = drawVerticalSegments(segV, finalImage.rows, finalImage.cols);
        
        // Create pairs
        vector<pair<int, int> > verticalPairs = filterVerticalPairs(createSegmentPairs(segV, finalImage.rows));
        vector<pair<int, int> > horizontalPairs = splitLarge(filterHorizontalPairs(createSegmentPairs(segH, finalImage.cols), finalImage.cols));
        
        // Get segment squares
        vector<Square> squares = takeSquares(shrinkSquares(finalImage, getSquares(verticalPairs, horizontalPairs)), 6);
        
        // Save the squares
        saveSquares(finalImage, squares, OUTPUT_PATH, captchaCode, counter);
        
        // Let's draw the rectangles
        drawSquares(finalImage, squares);
        drawSquares(sourceImage, squares);
        
        imshow("Final image", finalImage);
        imshow("Source image", sourceImage);
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
