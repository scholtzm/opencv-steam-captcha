#include <algorithm>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

bool compareContours(vector<Point> a, vector<Point> b) {
	return contourArea(a) > contourArea(b);
}

int main(int argc, char** argv) {
//	namedWindow("Window", CV_WINDOW_AUTOSIZE);

	const int BYTES_PER_PIXEL = 1;
	const string DATA_PATH = "/Users/Mike/Documents/Eclipse Workspace/Steam Captcha Breaker/Debug/data/";

	int total = 0;
	int success = 0;

	fs::path folder(DATA_PATH);

	if(!exists(folder))
		return -1;

	fs::directory_iterator endItr;
	for(fs::directory_iterator itr(folder); itr != endItr; itr++) {
		total++;

		string fullPath = itr->path().string();
		string fileName = itr->path().filename().string();

		string captchaCode = boost::replace_all_copy(fileName, ".png", "");
		boost::replace_all(captchaCode, "at", "@");
		boost::replace_all(captchaCode, "pct", "%");
		boost::replace_all(captchaCode, "and", "&");

		cout << total << ".) file: " << fileName << ", code: " << captchaCode;

		// Load our base image
		Mat sourceImage = imread(fullPath, CV_LOAD_IMAGE_GRAYSCALE);

		// Is it loaded correctly?
		if(!sourceImage.data)
			return -1;

		// Define our final image
		Mat finalImage;

		// Let's do some image operations
		threshold(sourceImage, finalImage, 150, 255, THRESH_BINARY);
		//adaptiveThreshold(sourceImage, finalImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 0);

		// Let's find contours
		vector<vector<Point> > contours;
		findContours(finalImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		// Sort them by size
		sort(contours.begin(), contours.end(), compareContours);

		// Draw 6 largest contours
		finalImage.setTo(Scalar(0));
		int contourCount = MIN(contours.size(), 6);
		for(int contourIndex = 0; contourIndex < contourCount; contourIndex++)
			drawContours(finalImage, contours, contourIndex, Scalar(255), CV_FILLED);

		// Initiate tesseract
		tesseract::TessBaseAPI tess;
		tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
		tess.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ@%&");
		tess.SetImage((uchar *) finalImage.data, finalImage.cols, finalImage.rows, BYTES_PER_PIXEL, BYTES_PER_PIXEL * finalImage.cols);

		string result(tess.GetUTF8Text());
		result.erase(remove_if(result.begin(), result.end(), ::isspace ), result.end());

		cout << ", result: " << result;

		if(captchaCode == result) {
			cout << " - TRUE" << endl;
			success++;
		} else {
			cout << " - FALSE" << endl;
		}

	//	imshow("Source", sourceImage);
	//	imshow("Final", finalImage);

	//	waitKey();

		sourceImage.release();
		finalImage.release();
		contours.clear();
	}

	cout << "Success rate: " << ((double)success/(double)total)*100 << "% (" << success << "/" << total << ")." << endl;

	return 0;
}
