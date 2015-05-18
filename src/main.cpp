#include <algorithm>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "imagereconstruct.hpp"
#include "image.hpp"
#include "segments.hpp"
#include "descriptors.hpp"
#include "misc.hpp"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
    const int RESIZE_FACTOR = 2;
    const string DATA_PATH = "/Users/Mike/Documents/Eclipse Workspace/SCB3/data/";
    const string OUTPUT_PATH = "/Users/Mike/Documents/Eclipse Workspace/SCB3/output/";
    
    // 0, 1, 5, 6, I, O and S are never used
    const string ALLOWED_CHARS = "234789ABCDEFGHJKLMNPQRTUVWXYZ@&%";
    
    // Images
    Mat sourceImage, finalImage, histogramImage;
    Mat histogram;
    
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
    if(!createFolderStructure(OUTPUT_PATH, ALLOWED_CHARS))
        return -1;
    
    fs::directory_iterator endItr;
    for(fs::directory_iterator itr(folder); itr != endItr; itr++) {
        string fullPath = itr->path().string();
        string fileName = itr->path().filename().string();
        
        // Skip all dot files
        if(fileName[0] == '.')
            continue;
        
        total++;
        
        // Retrieve captcha string
        string captchaCode = boost::replace_all_copy(fileName, ".png", "");
        captchaCode = aliasToSpecialChar(captchaCode);
        
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
        Mat tmp;
        sourceImage.copyTo(tmp, finalImage);
        tmp.copyTo(finalImage);
        tmp.release();
        
        // Normalize new image
        normalize(finalImage, finalImage, 0, 255, NORM_MINMAX, CV_8U);
        
        // Let's calculate histogram for our image
        histogram = createHistogram(finalImage);
        
        // Calculate final threshold value
        int thresholdValue = getIdealThreshold(histogram);
        
        // Draw histogram image
        histogramImage = drawHistogram(histogram, thresholdValue);
        
        // Apply binary threshold
        threshold(finalImage, finalImage, thresholdValue, 255, THRESH_BINARY);
        
        // Morphological closing
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
        vector<Rectangle> squares = takeRectangles(shrinkRectangles(finalImage, getRectangles(verticalPairs, horizontalPairs)), 6);
        
        // Save the squares
        
        // Let's draw the rectangles
        drawRectangles(finalImage, squares);
        drawRectangles(sourceImage, squares);
        
//        imshow("Final image", finalImage);
//        imshow("Source image", sourceImage);
////        imshow("HSeg", segHImage);
////        imshow("VSeg", segVImage);
//        imshow("Histogram", histogramImage);
//        waitKey();
        
        sourceImage.release();
        finalImage.release();
        
//        if(total == 20) break;
    }
    
    Mat trainingData, classLabels;
    getSimpleDescriptors(trainingData, classLabels, OUTPUT_PATH, "G", "Y", 25);
//    getHOGDescriptors(trainingData, classLabels, OUTPUT_PATH, "G", "Y", 25);
   
    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    
    CvSVM SVM;
    SVM.train(trainingData, classLabels, Mat(), Mat(), params);
    SVM.save((OUTPUT_PATH + "svm_data.dat").c_str());
    
    for(int i = 25; i < 29; i++) {
        Mat letterImage = imread(OUTPUT_PATH + "G/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        
        float *f = (float *) malloc(32 * 48 * sizeof(float));

        for(int j = 0; j < letterImage.rows; j++)
            for(int k = 0; k < letterImage.cols; k++)
                f[j * letterImage.cols + k] = ((float) letterImage.at<unsigned char>(j, k)) / 255.0;
        
        Mat testSample(1, 32 * 48, CV_32FC1, f);
        
        float result = SVM.predict(testSample);
        
        cout << "Predikcia: " << result << endl;
        
        free(f);
        letterImage.release();
    }
    
    
//    // HoG and SVM
//    vector<vector<float> > features;
//    vector<float> classes;
//    
//    int sampleSize = 25;
//    
//    for(int i = 0; i < sampleSize; i++) {
//        HOGDescriptor hog;
//        hog.winSize = Size(32, 48);
//        
//        vector<float> featureVector;
//        vector<float> featureVector2;
//        
//        Mat posImage = imread(OUTPUT_PATH + "G/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
//        if (!posImage.data)
//            break;
//        hog.compute(posImage, featureVector, Size(8, 8), Size(0, 0));
//        features.push_back(featureVector);
//        classes.push_back(1.0);
//        
//        Mat negImage = imread(OUTPUT_PATH + "Y/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
//        if (!negImage.data)
//            break;
//        hog.compute(negImage, featureVector2, Size(8, 8), Size(0, 0));
//        features.push_back(featureVector2);
//        classes.push_back(-1.0);
//    }
//    
//    float **featuresArray;
//    featuresArray = (float **) malloc(sampleSize * 2 * sizeof(float *));
//    for (int i = 0; i < sampleSize * 2; i++)
//        featuresArray[i] = (float *) malloc(540 *sizeof(float));
//    
//    for(int i = 0; i < features.size(); i++) {
//        for(int j = 0; j < features[i].size(); j++) {
//            featuresArray[i][j] = features[i][j];
//        }
//    }
//    
//    float *classesArray;
//    classesArray = (float *) malloc(sampleSize * 2 * sizeof(float));
//    for(int i = 0; i < sampleSize * 2; i++)
//        classesArray[i] = classes[i];
//    
//    for(int i = 0; i < features.size(); i++) {
//        for(int j = 0; j < features[i].size(); j++) {
//            if(featuresArray[i][j] != features[i][j])
//                cout << "FAIL" << endl;
//        }
//    }
//    
//    Mat trainingData(sampleSize * 2, 540, CV_32FC1, &featuresArray);
//    Mat classLabels(sampleSize * 2, 1, CV_32FC1, classesArray);
//    
//    CvSVMParams params;
//    params.svm_type = CvSVM::C_SVC;
//    params.kernel_type = CvSVM::LINEAR;
//    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
//
//    CvSVM SVM;
//    SVM.train(trainingData, classLabels, Mat(), Mat(), params);
//    SVM.save((OUTPUT_PATH + "svm_data.dat").c_str());
//    
//    // 4 positive and 4 negative samples
//    for(int i = 25; i < 29; i++) {
//        HOGDescriptor hog;
//        hog.winSize = Size(32, 48);
//        
//        Mat letterImage = imread(OUTPUT_PATH + "Y/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
//        
//        vector<float> featureVector;
//        hog.compute(letterImage, featureVector, Size(8, 8), Size(0, 0));
//        
//        float *f;
//        f = (float *) malloc(540 * sizeof(float));
//        for(int i = 0; i < 540; i++)
//            f[i] = featureVector[i];
//        
//        Mat testSample(1, 540, CV_32FC1, &f);
//        
//        float result = SVM.predict(testSample);
//        
//        cout << "Predikcia: " << result << endl;
//        
//        free(f);
//        featureVector.clear();
//        letterImage.release();
//    }
    
    return 0;
}
