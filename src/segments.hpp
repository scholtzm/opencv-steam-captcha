#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct Rectangle {
    int id; // position in the string
    int x;
    int y;
    int width;
    int height;
};

/**
 * Functions to create and draw segments.
 */
int* horizontalSegments(Mat& src);
int* verticalSegments(Mat& src);
Mat drawHorizontalSegments(int* seg, int rows, int cols);
Mat drawVerticalSegments(int* seg, int rows, int cols);

/**
 * Functions to manipulate segments.
 */
vector<pair<int, int> > createSegmentPairs(int* seg, int segSize);
vector<pair<int, int> > filterVerticalPairs(vector<pair<int, int> > verticalPairs);
vector<pair<int, int> > filterHorizontalPairs(vector<pair<int, int> > horizontalPairs, int segSize);
vector<pair<int, int> > splitLarge(vector<pair<int, int> > horizontalSegments);

/**
 * Functions to create, manipulate and draw rectangles around letters.
 */
vector<Rectangle> getRectangles(vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs);
vector<Rectangle> shrinkRectangles(Mat& image, vector<Rectangle> squares);
vector<Rectangle> takeRectangles(vector<Rectangle> squares, int number);
void drawRectangles(Mat& image, vector<Rectangle> squares);
void saveRectangles(Mat& image, vector<Rectangle> squares, string output_folder, string code, map<string, int>& counter);

/**
 * Obsolete: Use drawSquares instead.
 */
void drawSegmentRectangles(Mat& image, vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs);