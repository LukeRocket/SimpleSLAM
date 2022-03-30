#pragma once

#include <GL/glew.h>
#include <opencv2/core/mat.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/gl/gl.h>
#include <pangolin/handler/handler.h>

class PangolinRender
{
public:
	PangolinRender(int windWidth, int windHeight, int projPLaneWidth, int projPlaneHeight);
	~PangolinRender();

	void setCameraLocation(cv::Mat cameraCoords, float distanceFromKf = 10.f);
	void renderFrame(cv::Mat cvPointsPositions, std::vector<cv::Mat> KeyFramesPositions, std::vector<float> cols3DPoints = {});
	void drawKeyFrames(std::vector<cv::Mat> positions);
	void draw3DPoints(cv::Mat cvPositions, std::vector<float> cols, float squareDim = .1f);
	bool quitWindow();
private: 
	int windowWidth;
	int windowHeight;
	int projPlaneWidth;
	int projPlaneHeight;

	pangolin::View d_cam;
	pangolin::OpenGlRenderState* s_cam;
	pangolin::Handler3D* handler;
};

