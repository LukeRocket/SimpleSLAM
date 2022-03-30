#pragma once

#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/StdVector>

#include <opencv2/core/eigen.hpp>


class Slammer
{
public:
	Slammer(cv::Ptr<cv::ORB> extractor, cv::DescriptorMatcher* matcher, int nMatches = 100);
	

	void detect(cv::Mat frame);		
	std::vector<float> getColors(cv::Mat frame, cv::Mat pts);
	std::vector<cv::Mat> getMatchesPoints();
	std::vector<cv::KeyPoint> getKps();
	void match(cv::Mat frame);
	std::array<cv::Mat, 3> singularValueDecomp(cv::Mat matrix);
	void applyMaskToMatches(std::vector<cv::Mat>& matchesPts, cv::Mat mask);
	cv::Mat getCameraPoseMatrix(cv::Mat camMatrix, std::vector<cv::Mat>& matchesPts);
	void homogCoords(cv::Mat* coords);
	void normalizeCoords(cv::Mat* points, cv::Mat camMatrix);
	void denormalizeCoords(cv::Mat* points, cv::Mat camMatrix);
	cv::Mat getCameraProjectionMatrix(cv::Mat camPose, cv::Mat camMatrix);
	cv::Mat getCameraLocation(cv::Mat startLocations, cv::Mat camPose);
	cv::Mat getCurrentCameraTransform(cv::Mat newPose, cv::Mat oldPose);
	cv::Mat dehomogenizedCoords(cv::Mat points3DHomog);
	cv::Mat get3DPoints(std::vector<cv::Mat> matchesPts, cv::Mat oldProjectMatrix, cv::Mat projectMatrix);
	Eigen::Vector3f getFundamentalMatrixSVD(std::vector<cv::Mat> matchesPts = {});
	std::vector<cv::Mat> matchePts = {};
private:
	cv::Mat oldDescrs = {};
	cv::Mat newDescrs = {};
	std::vector<cv::KeyPoint> oldKps = {};
	std::vector<cv::KeyPoint> newKps = {};
	cv::Ptr<cv::ORB> featExtractor;
	cv::DescriptorMatcher* matcher;
	std::vector<cv::DMatch> matches = {};

	int maxNMatches;
}; 

