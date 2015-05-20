#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>

#include "segments.hpp"

using namespace std;
using namespace cv;

bool compareSquaresBySize(Rectangle a, Rectangle b) {
    return a.width * a.height > b.width * b.height;
}

bool compareSquaresById(Rectangle a, Rectangle b) {
    return a.id < b.id;
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
    const int MAGIC_SIZE = 14;
    
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

vector<Rectangle> getRectangles(vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs) {
    vector<Rectangle> squares;
    int id = 0;
    
    for(vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
        for(vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
            Rectangle r = { id++, itH->first, itV->first, itH->second - itH->first, itV->second - itV->first };
            squares.push_back(r);
        }
    }
    
    return squares;
}

vector<Rectangle> shrinkRectangles(Mat& image, vector<Rectangle> squares) {
    vector<Rectangle> new_squares;
    
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
        
        Rectangle r = { squares[i].id, squares[i].x + left, squares[i].y + top, right - left, bottom - top };
        new_squares.push_back(r);
        tmp.release();
        tmp2.release();
    }
    
    return new_squares;
}

vector<Rectangle> takeRectangles(vector<Rectangle> squares, int number) {
    vector<Rectangle> new_squares;
    
    sort(squares.begin(), squares.end(), compareSquaresBySize);
    int min = MIN(number, (int)squares.size());
    
    for(int i = 0; i < min; i++) {
        new_squares.push_back(squares[i]);
    }
    
    sort(new_squares.begin(), new_squares.end(), compareSquaresById);
    
    return new_squares;
}

void drawRectangles(Mat& image, vector<Rectangle> squares) {
    for(int i = 0; i < squares.size(); i++)
        rectangle(image, Point(squares[i].x, squares[i].y), Point(squares[i].x + squares[i].width, squares[i].y + squares[i].height), Scalar(255));
}

void saveRectangles(Mat& image, vector<Rectangle> squares, string output_folder, string code, map<string, int>& counter) {
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
        
        resize(tmp2, tmp2, Size(32, 48));
        
        imwrite(filename, tmp2);
        
        tmp.release();
        tmp2.release();
    }
}

void drawSegmentRectangles(Mat& image, vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs) {
    for(vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
        for(vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
            rectangle(image, Point(itH->first, itV->first), Point(itH->second, itV->second), Scalar(255));
        }
    }
}