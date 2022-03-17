/*
* #################################
* Author: Luca Sciarpa
* Date:   1/3/2022
* 
* Project:Simple Slam 
* #################################
  Notes:
    main external libs involved
	- CV2:		 image recognition	
    - Pangolin:  3D representation

    interesting resource https://docs.opencv.org/3.4/dc/d2c/tutorial_real_time_pose.html

  TODO :
    from 2d point to 3d 
        + study homography and fundamental matrix
        + find camera pose 
            - pnp ransac 
                > camera intrinsic matrix (focal lenght/center point location) 
                > 3d coord/2d coord relation of a set of points
                    * find pose matrix with fundamental matrix                                             
                        x = PX 
                        * x >> 2d point
                        * P >> projection matrix
                        * X >> 3d point                        
                        * 
*/

#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <GL/glew.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/StdVector>

#include <opencv2/core/eigen.hpp>

#include <pangolin/gl/gl.h>


//#define VIDEO_PATH 0
#define VIDEO_PATH "../src/190416_10_Drone1_01.mp4"


//#define FRAME_WIDTH 720
//#define FRAME_HEIGHT 360
#define FRAME_WIDTH (int)1920/2
#define FRAME_HEIGHT (int)1080/2


void orbExtraction(cv::Ptr<cv::ORB> orb, cv::Mat frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {        
    cv::Mat features = {};
    cv::Mat grayFrame = {};
    cv::cvtColor(frame, grayFrame, cv::COLOR_RGB2GRAY);
    cv::goodFeaturesToTrack(grayFrame, features, 3000, 0.01, 3.0);
    
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
    for (cv::Mat *matrix : matrices) {    
        matrix->convertTo(*matrix, CV_32F);    
    }
}

class PangolinRenderer {
public:    
    int height = 720;
    int width = 1280;

    PangolinRenderer() {   
        pangolin::CreateWindowAndBind("Main", width, height);
        glEnable(GL_DEPTH_TEST);

        s_cam = new pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(width, height, 420, 420, width/2.f, height/2.f, 0.2, 100),
            pangolin::ModelViewLookAt(0, 0, 1, 0, 0, 0, pangolin::AxisY)
        );

        handler  = new pangolin::Handler3D(*s_cam);       
        d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -(float)width / (float)height)
            .SetHandler(handler);
    }

    ~PangolinRenderer() {
        delete s_cam;
        delete handler;
    }

    void renderFrame(cv::Mat cvPointsPositions, cv::Mat trianglesPositions,
        std::vector<float> cols3DPoints = {}) {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); /// wireframe mode
        d_cam.Activate(*s_cam);                
                
        // draw points
        this->drawTriangles(trianglesPositions);
        this->draw3DPoints(cvPointsPositions, cols3DPoints);
        
        // Swap frames and Process Events
        pangolin::glDrawAxis(0.2f);
        pangolin::FinishFrame();       
    }       

    void drawTriangles(cv::Mat positions) {
        std::vector<float> cols = {};
        //  define points color if not already specified     
        if (cols.size() == 0) {
            for (int rIndex = 0; rIndex < positions.rows; rIndex++) {
                cols.push_back(1.f);
                cols.push_back(0.f);
                cols.push_back(0.f);
            }
        }

        std::vector<float> points = {};
        for (int rIndex = 0; rIndex < positions.rows; rIndex++) {
            points.push_back(positions.at<float>(rIndex, 0));
            points.push_back(positions.at<float>(rIndex, 1));
            points.push_back(positions.at<float>(rIndex, 2));
        }
        pangolin::glDrawColoredVertices<float, float>((int)points.size() / 3, points.data(), cols.data(), GL_TRIANGLE_FAN, 3, 3);
    }

    void draw3DPoints(cv::Mat cvPositions, std::vector<float> cols) {
        std::vector<float> point = {};
        std::vector<float> pointCol = {};

        for (int rIndex = 0; rIndex < cvPositions.rows; rIndex++) {            
            point.push_back(cvPositions.at<float>(rIndex, 0) - 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 1) - 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 2));

            point.push_back(cvPositions.at<float>(rIndex, 0) - 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 1) + 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 2));

            point.push_back(cvPositions.at<float>(rIndex, 0) + 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 1) + 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 2));
            
            point.push_back(cvPositions.at<float>(rIndex, 0) + 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 1) - 0.01f);
            point.push_back(cvPositions.at<float>(rIndex, 2));


            for (int colIter = 0; colIter < 3; colIter++) {
                pointCol.push_back(cols[rIndex * 3]/255);
                pointCol.push_back(cols[rIndex * 3 + 1]/255);
                pointCol.push_back(cols[rIndex * 3 + 2]/255);
            }            
            pangolin::glDrawColoredVertices<float, float>((int)4, point.data(), pointCol.data(), GL_TRIANGLE_FAN, 3, 3);                    
            
            point.clear();
            pointCol.clear();
        }        
     
    }

public:
    pangolin::View d_cam;
    pangolin::OpenGlRenderState* s_cam;
    pangolin::Handler3D* handler;
};



class Slammer {
public:
    Slammer(cv::Ptr<cv::ORB> extractor, cv::DescriptorMatcher * matcher, int nMatches = 100){
        this->featExtractor = extractor;
        this->matcher = matcher;        
        this->maxNMatches = nMatches;
    }

    void detect(cv::Mat frame) {
        // back end 
        this->oldDescrs = this->newDescrs.clone();
        this->oldKps = this->newKps;

        orbExtraction(this->featExtractor, frame, this->newKps, this->newDescrs);                   

        this->match(frame);
    }

    std::vector<float> getColors(cv::Mat frame, cv::Mat pts) {
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
    
    std::vector<cv::Mat> getMatchesPoints() {
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

    std::vector<cv::KeyPoint> getKps() {
        return this->newKps;
    }

    void match(cv::Mat frame) {        
        if (!this->oldDescrs.empty()) {
            std::vector<std::vector<cv::DMatch>> matchesList = {};
                                       
            this->matches.clear();
            this->matcher->knnMatch(this->newDescrs, this->oldDescrs, matchesList, 2);

            for (int i = 0; i < matchesList.size(); i++) {
                if (matchesList[i][0].distance < 0.75 * matchesList[i][1].distance) {
                  this->matches.push_back(matchesList[i][0]);                
                }
            }            
            
            /*struct {
                bool operator()(cv::DMatch a, cv::DMatch b) const { return a.distance < b.distance; }
            } distanceLess;

            std::sort(this->matches.begin(), this->matches.end(), distanceLess);
            if (this->matches.size() >= 100) {
                this->matches.resize(this->maxNMatches);
            }*/
        }
    }

    cv::Mat getCameraPoseMatrix(cv::Mat camMatrix, std::vector<cv::Mat> matchesPts = {}) {

        cv::Mat mask = {};
        
        
        cv::Mat E = cv::findFundamentalMat(matchesPts[1], matchesPts[0], mask, cv::FM_RANSAC, (double).005f);

        //camMatrix.at<float>(0, 2) = 0;
        //camMatrix.at<float>(1, 2) = 0;
        //camMatrix.at<float>(1, 1) = 1;
        //camMatrix.at<float>(0, 0) = 1;
        
        //std::vector<cv::Point2f> coords = {};
        //std::vector<cv::Point2f> coords1 = {};

        //for (int rIndex = 0; rIndex < matchesPts[0].rows; rIndex++) {
        //    coords.push_back(cv::Point2f(matchesPts[0].at<float>(rIndex, 0), matchesPts[0].at<float>(rIndex, 1)));
        //    coords1.push_back(cv::Point2f(matchesPts[1].at<float>(rIndex, 0), matchesPts[1].at<float>(rIndex, 1)));
        //}


        //// prob | threshold | maxIters 
        ////cv::Mat E = cv::findEssentialMat(matchesPts[0], matchesPts[1], camMatrix, cv::RANSAC, 0.99f, .005f, 200, mask);
        //cv::Mat E = cv::findEssentialMat(coords, coords1, camMatrix, cv::RANSAC, 0.99f, .005f, 200, mask);
        matchesPts[0].copyTo(matchesPts[0], mask);
        matchesPts[1].copyTo(matchesPts[1], mask);

        // svd of the Essential matrix 
        Eigen::Matrix3f FEigen;
        cv::cv2eigen(E, FEigen);
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(FEigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
        std::cout << svd.singularValues() << std::endl;

        cv::Mat R = {};
        cv::Mat R2 = {};
        cv::Mat t = {};
        cv::Mat cameraPoseMatrix;

        //cv::recoverPose(E, matchesPts[0], matchesPts[1], camMatrix, R, t, mask);        
        cv::decomposeEssentialMat(E, R, R2, t);

        /*matchesPts[0].copyTo(matchesPts[0], mask);
        matchesPts[1].copyTo(matchesPts[1], mask);*/

        cv::hconcat(R, t, cameraPoseMatrix);
        return cameraPoseMatrix;        
    }

    void homogCoords(cv::Mat *coords) {             
        *coords = coords->reshape(1);        
        cv::hconcat(*coords, cv::Mat(coords->rows, 1, CV_32F, 1.f), *coords);                
    }

    void normalizeCoords(cv::Mat *points, cv::Mat camMatrix) {
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

    void denormalizeCoords(cv::Mat* points, cv::Mat camMatrix) {
        this->homogCoords(points);
        std::vector<cv::Mat*> m = { points, &camMatrix };
        standardizeMatricesType(m);

        *points = (camMatrix * points->t()).t();

        cv::Mat finalPoints = {};
        for (int rIndex = 0; rIndex < points->rows; rIndex++)
            finalPoints.push_back(cv::Point2f(points->at<float>(rIndex, 0), points->at<float>(rIndex, 1)));
        *points = finalPoints.clone();
    }


    cv::Mat getCameraProjectionMatrix(cv::Mat camPose, cv::Mat camMatrix) {
        std::vector<cv::Mat*> m = {&camMatrix, &camPose};
        standardizeMatricesType(m);
        return camMatrix * camPose;
    }

    cv::Mat getCameraLocation(cv::Mat startLocations, cv::Mat camPose) {
        std::vector<float> lastRowData = { 0, 0, 0, 1 };
        cv::Mat lastRow = cv::Mat(1, 4, CV_32F, lastRowData.data());
        camPose.convertTo(camPose, CV_32F);
        cv::vconcat(camPose, lastRow, camPose);

        cv::Mat endLocations;
        std::vector<cv::Mat*> matrices = {&camPose, &startLocations };
        standardizeMatricesType(matrices);        


        for (int rIndex = 0; rIndex < startLocations.rows; rIndex++) {            
            /*endLocations.push_back((camMatrix * camPose * startLocations.row(rIndex).t()).t());*/
            endLocations.push_back((camPose * startLocations.row(rIndex).t()).t());
        }
        return endLocations;
    }   


    cv::Mat dehomogenizedCoords(cv::Mat points3DHomog) {
        cv::Mat point3dDehomog = {};
        if (points3DHomog.cols != 4)  return point3dDehomog;
        for (int rIndex = 0; rIndex < points3DHomog.rows; rIndex ++) {
            cv::Mat row = points3DHomog.row(rIndex);
            for (int cIndex = 0; cIndex < 3; cIndex++)
                row.at<float>(0, cIndex) /= row.at<float>(0, 3);
            row(cv::Range(0, 1), cv::Range(0, 3)).copyTo(row);
            point3dDehomog.push_back(row);
        }
        return point3dDehomog;
    }

    cv::Mat get3DPoints(std::vector<cv::Mat> matchesPts, cv::Mat oldProjectMatrix, cv::Mat projectMatrix) {
       
        cv::Mat points3D = {};       
        cv::triangulatePoints(oldProjectMatrix, projectMatrix, matchesPts[0].t(), matchesPts[1].t(), points3D);
        points3D = this->dehomogenizedCoords(points3D.t());
        
        return points3D;
    }
    
    float estimateCameraFocalLenght(std::vector<float> v){    
        return median(v);
    }


    Eigen::Vector3f getFundamentalMatrixSVD(std::vector<cv::Mat> matchesPts = {}) {               
        if(matchesPts[0].empty() == true)
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

    std::vector<cv::Mat> matchesPts = {};

private:
    cv::Mat oldDescrs = {};
    cv::Mat newDescrs = {};
    std::vector<cv::KeyPoint> oldKps = {};
    std::vector<cv::KeyPoint> newKps = {};
    cv::Ptr<cv::ORB> featExtractor;
    cv::DescriptorMatcher *matcher;
    std::vector<cv::DMatch> matches = {};
    
    int maxNMatches;
};


int main()
{  
    // feature extractor
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    // matcher 
    //cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1); // instantiate LSH index parameters
    //cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);       // instantiate flann search parameters
    //cv::DescriptorMatcher* matcher = new cv::FlannBasedMatcher(indexParams, searchParams);

    cv::DescriptorMatcher* matcher = new cv::BFMatcher();
    Slammer* slam = new Slammer(orb, matcher);
    
    // video aquisition 
    cv::VideoCapture cap(VIDEO_PATH);
    cv::Mat frame;
   
    // pangolin settings
    PangolinRenderer* pr = new PangolinRenderer();

    int counter = 0;
    std::vector<float> mCollection = {};
    float focalLenght = 200;
    // define camera matrix
    cv::Mat camMatrix = cv::Mat::zeros(3, 3, CV_32F);

    // set camera center 
    camMatrix.at<float>(0, 2) += (int)(FRAME_WIDTH / 2);
    camMatrix.at<float>(1, 2) += (int)(FRAME_HEIGHT / 2);

    // set focal lenght
    camMatrix.at<float>(0, 0) += (int)(focalLenght);
    camMatrix.at<float>(2, 2) += 1.f;   
    camMatrix.at<float>(1, 1) += (int)(focalLenght);

    cv::Mat oldP;
    cv::Mat points3D = {};
    // normalized w.r.t. frame width (considered equal to 1)    
    std::vector<float> locationData = { -0.5f, -FRAME_HEIGHT / (float)(2 * FRAME_WIDTH) , 0.f, 1.f,
                                        -0.5f,  FRAME_HEIGHT / (float)(2 * FRAME_WIDTH) , 0.f, 1.f,
                                         0.5f,  FRAME_HEIGHT / (float)(2 * FRAME_WIDTH) , 0.f, 1.f,
                                         0.5f, -FRAME_HEIGHT / (float)(2 * FRAME_WIDTH), 0.f, 1.f,
                                         0.f, 0.f, camMatrix.at<float>(0, 0)/(float)FRAME_WIDTH, 1.f};
    cv::Mat startLocationVertices = cv::Mat(5, 4, CV_32F, locationData.data());

    while (cap.isOpened()) {                    
        cap >> frame;                
        if (counter % 1 == 0) {
            if (!frame.empty()) {
                cv::resize(frame, frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
                slam->detect(frame);                     
                std::vector<cv::Mat> matchesPts = slam->getMatchesPoints();
                                             
               /* if (!matchesPts[0].empty()) {                                        
                    slam->normalizeCoords(&matchesPts[0], camMatrix);
                    slam->normalizeCoords(&matchesPts[1], camMatrix);

                    cv::Mat cameraPose = slam->getCameraPoseMatrix(camMatrix, matchesPts);
                    cv::Mat endLocationVertices = slam->getCameraLocation(startLocationVertices, cameraPose);                    
                    cv::Mat cameraProj = slam->getCameraProjectionMatrix(cameraPose, camMatrix);
                    if (oldP.empty() == false) {
                        cv::Mat pInFrame = slam->get3DPoints(matchesPts, oldP, cameraProj);
                        for (int rIndex = 0; rIndex < 3; rIndex++) {
                            points3D.push_back(pInFrame.row(rIndex));
                        }
                        
                        slam->denormalizeCoords(&matchesPts[1], camMatrix);
                        slam->denormalizeCoords(&matchesPts[0], camMatrix);
                        pr->renderFrame(points3D, endLocationVertices, slam->getColors(frame, matchesPts[1]));
                    }                    
                    oldP = cameraProj.clone();
                }
                */
                cv::drawKeypoints(frame, slam->getKps(), frame, cv::Scalar(0, 255, 0));
                for (int i = 0; i < matchesPts[0].rows; i++) {
                    cv::Point2f p1 = cv::Point2f(matchesPts[0].at<float>(i, 0), matchesPts[0].at<float>(i, 1));
                    cv::Point2f p2 = cv::Point2f(matchesPts[1].at<float>(i, 0), matchesPts[1].at<float>(i, 1));
                    cv::line(frame,
                        p1,
                        p2,
                        cv::Scalar(255, 0, 0), 1);
                }

                cv::imshow("Frame", frame);
                char c = (char)cv::waitKey(25);
                if (c == 27)
                    break;
            }
            else {
                break;
            }
        }
        counter += 1;
    }

    delete slam;
    delete matcher;    
    delete pr;
    
    cap.release();
    cv::destroyAllWindows();
  
    return 0;
}
