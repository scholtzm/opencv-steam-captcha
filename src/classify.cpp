#include <opencv2/opencv.hpp>

#include "main.hpp"
#include "descriptors.hpp"

using namespace cv;
using namespace std;

float classify(CvSVM& SVM, Mat& image) {
    Mat descriptor;
    
#if SIMPLE_DESCRIPTORS == 1
    getSimpleDescriptor(image, descriptor);
#else
    getHOGDescriptor(image, descriptor);
#endif
    
    float result = SVM.predict(descriptor);
    
    descriptor.release();
    
    return result;
}