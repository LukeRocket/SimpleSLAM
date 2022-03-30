#include "pangolinRender.h"


PangolinRender::PangolinRender(int windWidth, int windHeight, int projPLaneWidth, int projPlaneHeight) :
    windowWidth(windWidth), windowHeight(windHeight), projPlaneWidth(projPLaneWidth), projPlaneHeight(projPlaneHeight)
{ 
    pangolin::CreateWindowAndBind("Main", this->windowWidth, this->windowHeight);
    glEnable(GL_DEPTH_TEST);

    this->s_cam = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(this->projPlaneWidth, this->projPlaneHeight, 420, 420, this->projPlaneWidth / 2.f, this->projPlaneHeight / 2.f, 0.2, 100000),
        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY)
    );

    this->handler = new pangolin::Handler3D(*this->s_cam);
    this->d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -(float)this->windowWidth / (float)this->windowHeight)
        .SetHandler(this->handler);
}


PangolinRender::~PangolinRender() {
	delete this->s_cam;
	delete this->handler;
}


void PangolinRender::setCameraLocation(cv::Mat cameraCoords, float distanceFromKf) {
    std::vector<float> lookAtVertex = { 0,0,0 };
    for (int i = 0; i < 4; i++) {
        lookAtVertex[0] += cameraCoords.at<float>(i, 0) * 0.25f;
        lookAtVertex[1] += cameraCoords.at<float>(i, 1) * 0.25f;
        lookAtVertex[2] += cameraCoords.at<float>(i, 2) * 0.25f;
    }

    std::vector<float> cameraCenterLocation = { cameraCoords.at<float>(cameraCoords.rows - 1, 0),
        cameraCoords.at<float>(cameraCoords.rows - 1, 1),
        cameraCoords.at<float>(cameraCoords.rows - 1, 2) };

    this->s_cam = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(this->projPlaneWidth, this->projPlaneHeight, 420, 420, this->projPlaneWidth / 2.f, this->projPlaneHeight / 2.f, 0.2, 10000),
        pangolin::ModelViewLookAt(
            //0,0,0,
            cameraCenterLocation[0], cameraCenterLocation[1] + distanceFromKf, cameraCenterLocation[2] + distanceFromKf,
            lookAtVertex[0], lookAtVertex[1], lookAtVertex[2], pangolin::AxisY)
    );


    this->handler = new pangolin::Handler3D(*this->s_cam);
    this->d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -(float)this->windowWidth / (float)this->windowHeight)
        .SetHandler(this->handler);
}

void PangolinRender::renderFrame(cv::Mat cvPointsPositions, std::vector<cv::Mat> KeyFramesPositions, std::vector<float> cols3DPoints) {
    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); /// wireframe mode

    if (KeyFramesPositions.empty() != true) {
        this->setCameraLocation(KeyFramesPositions[KeyFramesPositions.size() - 1], 15.f);
    }

    this->d_cam.Activate(*this->s_cam);

    // draw points
    this->drawKeyFrames(KeyFramesPositions);
    this->draw3DPoints(cvPointsPositions, cols3DPoints);

    // Swap frames and Process Events
    pangolin::glDrawAxis(0.2f);
    pangolin::FinishFrame();
}

void PangolinRender::drawKeyFrames(std::vector<cv::Mat> positions) {
    for (cv::Mat p : positions) {
        std::vector<float> cols = {};
        //  define points color if not already specified     
        if (cols.size() == 0) {
            for (int rIndex = 0; rIndex < p.rows; rIndex++) {
                cols.push_back(1.f);
                cols.push_back(0.f);
                cols.push_back(0.f);
            }
        }

        std::vector<float> points = {};
        for (int rIndex = 0; rIndex < p.rows; rIndex++) {
            points.push_back(p.at<float>(rIndex, 0));
            points.push_back(p.at<float>(rIndex, 1));
            points.push_back(p.at<float>(rIndex, 2));
        }
        pangolin::glDrawColoredVertices<float, float>((int)points.size() / 3, points.data(), cols.data(), GL_TRIANGLE_FAN, 3, 3);
    }
}

void PangolinRender::draw3DPoints(cv::Mat cvPositions, std::vector<float> cols, float squareDim) {
    std::vector<float> point = {};
    std::vector<float> pointCol = {};

    for (int rIndex = 0; rIndex < cvPositions.rows; rIndex++) {
        point.push_back(cvPositions.at<float>(rIndex, 0) - squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 1) - squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 2));

        point.push_back(cvPositions.at<float>(rIndex, 0) - squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 1) + squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 2));

        point.push_back(cvPositions.at<float>(rIndex, 0) + squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 1) + squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 2));

        point.push_back(cvPositions.at<float>(rIndex, 0) + squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 1) - squareDim);
        point.push_back(cvPositions.at<float>(rIndex, 2));


        for (int colIter = 0; colIter < 3; colIter++) {
            pointCol.push_back(cols[rIndex * 3] / 255);
            pointCol.push_back(cols[rIndex * 3 + 1] / 255);
            pointCol.push_back(cols[rIndex * 3 + 2] / 255);
        }
        pangolin::glDrawColoredVertices<float, float>((int)4, point.data(), pointCol.data(), GL_TRIANGLE_FAN, 3, 3);
       
        point.clear();
        pointCol.clear();
    }
}

bool PangolinRender::quitWindow() {    
    std::cout << pangolin::ShouldQuit() << std::endl;
    return pangolin::ShouldQuit();
}