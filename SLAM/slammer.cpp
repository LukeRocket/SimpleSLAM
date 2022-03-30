#include "slammer.h"
#include <opencv2/imgproc.hpp>


typedef uint8_t Pixel;

void orbExtraction(cv::Ptr<cv::ORB> orb, cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    cv::Mat meanFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    meanFrame.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
        cv::Vec3b p = frame.at<cv::Vec3b>(position[0], position[1]);
        pixel = (Pixel)(p[0] + p[1] + p[2]);
        });

    cv::Mat features;
    cv::goodFeaturesToTrack(meanFrame, features, 3000, 0.01, 3);

    keypoints = {};
    for (int rIndex = 0; rIndex < features.rows; rIndex++) {
        keypoints.push_back(cv::KeyPoint(cv::Point2f(features.at<float>(rIndex, 0), features.at<float>(rIndex, 1)), 2));
    }

    orb->compute(frame, keypoints, descriptors);
}

float median(std::vector<float> vec) {
    int nMiddle = (int)(vec.size() / 2) + 1;
    std::nth_element(vec.begin(), vec.begin() + nMiddle, vec.end());
    return vec[nMiddle - 1];
}

void standardizeMatricesType(std::vector<cv::Mat*> matrices) {
    for (cv::Mat* matrix : matrices) {
        matrix->convertTo(*matrix, CV_32F);
    }
}


Slammer::Slammer(cv::Ptr<cv::ORB> extractor, cv::DescriptorMatcher * matcher, int nMatches) {
    this->featExtractor = extractor;
    this->matcher = matcher;
    this->maxNMatches = nMatches;
}

void Slammer::detect(cv::Mat frame) {
    // back end 
    this->oldDescrs = this->newDescrs.clone();
    this->oldKps = this->newKps;

    orbExtraction(this->featExtractor, frame, this->newKps, this->newDescrs);

    this->match(frame);
}


std::vector<float> Slammer::getColors(cv::Mat frame, cv::Mat pts) {
    std::vector<float> colors = {};
    // extract informations
    for (int rIndex = 0; rIndex < pts.rows; rIndex++) {
        //add colors  
        cv::Point2f point = cv::Point2f(pts.at<float>(rIndex, 0), pts.at<float>(rIndex, 1));
        colors.push_back(frame.at<cv::Vec3b>(point)[0]);
        colors.push_back(frame.at<cv::Vec3b>(point)[1]);
        colors.push_back(frame.at<cv::Vec3b>(point)[2]);
    }
    return colors;
}

std::vector<cv::Mat> Slammer::getMatchesPoints() {
    std::vector<cv::Mat> matchPoints = { {}, {} };

    // extract informations        
    for (int i = 0; i < matches.size(); i++) {
        cv::Point2f p1;
        cv::Point2f p2;
        p1 = this->oldKps[std::min((int)matches[i].trainIdx, (int)this->oldKps.size() - 1)].pt;
        p2 = this->newKps[std::min((int)matches[i].queryIdx, (int)this->newKps.size() - 1)].pt;

        matchPoints[0].push_back(p1);
        matchPoints[1].push_back(p2);
    }

    return matchPoints;
}

std::vector<cv::KeyPoint> Slammer::getKps() {
    return this->newKps;
}

void Slammer::match(cv::Mat frame) {

    if (!this->oldDescrs.empty()) {
        std::vector<std::vector<cv::DMatch>> matchesList = {};

        this->matches.clear();
        this->matcher->knnMatch(this->newDescrs, this->oldDescrs, matchesList, 2);

        for (int i = 0; i < matchesList.size(); i++) {
            if (matchesList[i][0].distance < 0.65f * matchesList[i][1].distance) {
                this->matches.push_back(matchesList[i][0]);
            }
        }
    }
}

std::array<cv::Mat, 3> Slammer::singularValueDecomp(cv::Mat matrix) {
    /*
        Return array composed by:
            0: SingularValueMat
            1: right eigenVectsMat
            2: left eigenVectsMat
    */
    cv::Mat sigma = {};
    cv::Mat u = {};
    cv::Mat v = {};

    Eigen::Matrix3f eigenMat;
    cv::cv2eigen(matrix, eigenMat);
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(eigenMat, Eigen::ComputeFullU | Eigen::ComputeFullV);

    cv::eigen2cv(svd.singularValues(), sigma);
    cv::eigen2cv(svd.matrixU(), u);
    cv::eigen2cv(svd.matrixV(), v);

    std::array<cv::Mat, 3> output = { sigma, u, v };
    return output;
}


void Slammer::applyMaskToMatches(std::vector<cv::Mat> &matchesPts, cv::Mat mask) {

    cv::Mat out1 = {};
    cv::Mat out2 = {};

    for (int rIndex = 0; rIndex < mask.rows; rIndex++) {
        if (mask.at<uchar>(rIndex, 0) != 0) {
            out1.push_back(matchesPts[0].at<cv::Point2f>(rIndex, 0));
            out2.push_back(matchesPts[1].at<cv::Point2f>(rIndex, 0));
        }
    }

    matchesPts[0] = out1.clone();
    matchesPts[1] = out2.clone();
}

cv::Mat Slammer::getCameraPoseMatrix(cv::Mat camMatrix, std::vector<cv::Mat> &matchesPts) {
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(matchesPts[0].t(), matchesPts[1].t(), camMatrix, cv::RANSAC, 0.99f, 3.f, 1000, mask);

    this->applyMaskToMatches(matchesPts, mask);

    cv::Mat R = {};
    cv::Mat t = {};
    cv::Mat cameraPoseMatrix = {};

    mask = {};
    cv::recoverPose(E, matchesPts[0], matchesPts[1], camMatrix, R, t, mask);

    this->applyMaskToMatches(matchesPts, mask);

    cv::hconcat(R, t, cameraPoseMatrix);
    return cameraPoseMatrix;
}

void Slammer::homogCoords(cv::Mat * coords) {
    *coords = coords->reshape(1);
    cv::hconcat(*coords, cv::Mat(coords->rows, 1, CV_32F, 1.f), *coords);
}

void Slammer::normalizeCoords(cv::Mat * points, cv::Mat camMatrix) {
    this->homogCoords(points);
    cv::Mat KInv = camMatrix.inv();
    std::vector<cv::Mat*> m = { points, &KInv };
    standardizeMatricesType(m);

    *points = (KInv * points->t()).t();

    cv::Mat finalPoints = {};
    for (int rIndex = 0; rIndex < points->rows; rIndex++)
        finalPoints.push_back(cv::Point2f(points->at<float>(rIndex, 0), points->at<float>(rIndex, 1)));
    *points = finalPoints.clone();
}

void Slammer::denormalizeCoords(cv::Mat * points, cv::Mat camMatrix) {
    this->homogCoords(points);
    std::vector<cv::Mat*> m = { points, &camMatrix };
    standardizeMatricesType(m);

    *points = (camMatrix * points->t()).t();

    cv::Mat finalPoints = {};
    for (int rIndex = 0; rIndex < points->rows; rIndex++)
        finalPoints.push_back(cv::Point2f(points->at<float>(rIndex, 0), points->at<float>(rIndex, 1)));
    *points = finalPoints.clone();
}


cv::Mat Slammer::getCameraProjectionMatrix(cv::Mat camPose, cv::Mat camMatrix) {
    std::vector<cv::Mat*> m = { &camMatrix, &camPose };
    standardizeMatricesType(m);
    return camMatrix * camPose;
}

cv::Mat Slammer::getCameraLocation(cv::Mat startLocations, cv::Mat camPose) {
    std::vector<float> lastRowData = { 0, 0, 0, 1 };
    cv::Mat lastRow = cv::Mat(1, 4, CV_32F, lastRowData.data());
    camPose.convertTo(camPose, CV_32F);
    cv::vconcat(camPose, lastRow, camPose);

    cv::Mat endLocations;
    std::vector<cv::Mat*> matrices = { &camPose, &startLocations };
    standardizeMatricesType(matrices);


    for (int rIndex = 0; rIndex < startLocations.rows; rIndex++) {
        endLocations.push_back((camPose * startLocations.row(rIndex).t()).t());
    }

    return endLocations;
}

cv::Mat Slammer::getCurrentCameraTransform(cv::Mat newPose, cv::Mat oldPose) {
    std::vector<float> lastRowData = { 0, 0, 0, 1 };
    cv::Mat lastRow = cv::Mat(1, 4, CV_32F, lastRowData.data());

    newPose.convertTo(newPose, CV_32F);
    cv::vconcat(newPose, lastRow, newPose);
    oldPose.convertTo(oldPose, CV_32F);
    cv::vconcat(oldPose, lastRow, oldPose);

    cv::Mat cameraTransform = newPose * oldPose;
    return cameraTransform.rowRange(0, 3);
}


cv::Mat Slammer::dehomogenizedCoords(cv::Mat points3DHomog) {
    cv::Mat point3dDehomog = {};

    assert(points3DHomog.cols == 4);

    for (int rIndex = 0; rIndex < points3DHomog.rows; rIndex++) {
        cv::Mat row = points3DHomog.row(rIndex);
        for (int cIndex = 0; cIndex < 3; cIndex++)
            row.at<float>(0, cIndex) /= -row.at<float>(0, 3);
        row(cv::Range(0, 1), cv::Range(0, 3)).copyTo(row);
        point3dDehomog.push_back(row);
    }
    return point3dDehomog;
}

cv::Mat Slammer::get3DPoints(std::vector<cv::Mat> matchesPts, cv::Mat oldProjectMatrix, cv::Mat projectMatrix) {
    cv::Mat points3D = {};
    cv::triangulatePoints(projectMatrix, oldProjectMatrix, matchesPts[1].t(), matchesPts[0].t(), points3D);
    return this->dehomogenizedCoords(points3D.t());
}


Eigen::Vector3f Slammer::getFundamentalMatrixSVD(std::vector<cv::Mat> matchesPts) {
    if (matchesPts[0].empty() == true)
        matchesPts = this->getMatchesPoints();
    if (matchesPts[0].rows == 0) {
        Eigen::Vector3f vec = {};
        return vec;
    }

    cv::Mat mask = {};
    cv::Mat F = cv::findFundamentalMat(matchesPts[0], matchesPts[1], mask, cv::FM_RANSAC);

    for (cv::Mat p : matchesPts) {
        p.copyTo(p, mask);
    }

    Eigen::Matrix3f FEigen;
    cv::cv2eigen(F, FEigen);
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(FEigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.singularValues();
}
