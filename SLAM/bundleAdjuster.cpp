#include "bundleAdjuster.h"

std::vector<float> BundleAdjuster::getErrors() {
    return std::vector<float>({});
}

cv::Mat BundleAdjuster::adjust() {
    this->getErrors();
    return cv::Mat(2, 2, CV_32F);
}
