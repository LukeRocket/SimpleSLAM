#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>

class BundleAdjuster {
public:
    BundleAdjuster();
    ~BundleAdjuster();
    cv::Mat adjust();
    std::vector<float> getErrors();
private:
    cv::Mat CamMatrix;
    cv::Mat keyFrameLocation;
    cv::Mat pointsMapped;
};