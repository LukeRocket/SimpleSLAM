


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

#include "pangolinRender.h" 
#include "slammer.h"


//#define VIDEO_PATH 0
#define VIDEO_PATH "../src/190416_10_Drone1_01.mp4"
//#define VIDEO_PATH "../src/test.mp4"

#define FRAME_WIDTH (int)1920/2
#define FRAME_HEIGHT (int)1080/2
#define FOCAL_LENGTH (float)270.f

int main()
{  
    // feature extractor
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    // matcher     
    cv::DescriptorMatcher* matcher = new cv::BFMatcher();
    Slammer* slam = new Slammer(orb, matcher);
    
    // video aquisition 
    cv::VideoCapture cap(VIDEO_PATH);
    cv::Mat frame;
   
    // pangolin settings
    PangolinRender* pr = new PangolinRender(1280, 720, FRAME_WIDTH, FRAME_HEIGHT);

    int counter = 0;
    std::vector<float> mCollection = {};
   

    std::vector<float> camData = { FOCAL_LENGTH, 0, FRAME_WIDTH / 2,  0, FOCAL_LENGTH, FRAME_HEIGHT / 2, 0, 0, 1 };
    cv::Mat camMatrix = cv::Mat(3, 3, CV_32F, camData.data());

    // init keyframe location
    std::vector<float> cameraLocationData = { -0.5, -0.5f * (float)(FRAME_HEIGHT) / (float)(FRAME_WIDTH), -camMatrix.at<float>(0, 0) / (float)(FRAME_WIDTH), 1.f,
                                        -0.5f, 0.5f * (float)(FRAME_HEIGHT) / (float)(FRAME_WIDTH), -camMatrix.at<float>(0, 0) / (float)(FRAME_WIDTH), 1.f,
                                         0.5f, 0.5f * (float)(FRAME_HEIGHT) / (float)(FRAME_WIDTH), -camMatrix.at<float>(0, 0) / (float)(FRAME_WIDTH), 1.f,
                                         0.5f, -0.5f * (float)(FRAME_HEIGHT) / (float)(FRAME_WIDTH), -camMatrix.at<float>(0, 0) / (float)(FRAME_WIDTH), 1.f,
                                         0.f, 0.f, 0.f, 1.f }; // camera center point

    std::vector<cv::Mat> cameraLocations = { cv::Mat(5, 4, CV_32F, cameraLocationData.data()) };
    cv::Mat oldP = {};
    cv::Mat cameraProj = {};
    cv::Mat points3D = {};
    std::vector<float> points3DColor = {};
    cv::Mat oldCameraPose = {};
    cv::Mat newCameraTransform = {};

    while (cap.isOpened()) {                    
        cap >> frame;                
        if (!frame.empty() && counter % 2 == 0) {
            cv::resize(frame, frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
            // feature detection            
            slam->detect(frame);                     
            std::vector<cv::Mat> matchesPts = slam->getMatchesPoints();
                                             
            if (!matchesPts[0].empty()) {     // some match found                                                                     
                cv::Mat cameraPose = slam->getCameraPoseMatrix(camMatrix, matchesPts);
                               
                cv::Mat newCamLocation = slam->getCameraLocation(cameraLocations[cameraLocations.size() - 1], cameraPose);
                cameraLocations.push_back(newCamLocation);

                if (oldCameraPose.empty() == false) {
                    cameraPose = slam->getCurrentCameraTransform(cameraPose, oldCameraPose);                        
                }
                oldCameraPose = cameraPose.clone();

                cameraProj = slam->getCameraProjectionMatrix(cameraPose, camMatrix);                

                if (oldP.empty() == false) {                    
                    cv::Mat pInFrame = slam->get3DPoints(matchesPts, oldP, cameraProj);

                    //  https://answers.opencv.org/question/198880/recoverpose-and-triangulatepoints-3d-results-are-incorrect/
                    cv::Mat cameraCenterPoint = newCamLocation.row(newCamLocation.rows - 1).colRange(0, 3);

                    /*                    
                    * TODOs:    
                    *               >> perform bundle adjustment
                    */

                    for (int rIndex = 0; rIndex < pInFrame.rows; rIndex++) {                    
                        /// filter out results too far from camera center 
                        if(cv::norm(pInFrame.row(rIndex) - cameraCenterPoint) < 50) 
                            points3D.push_back(pInFrame.row(rIndex));
                    }                        

                    std::vector<float> frameColor = slam->getColors(frame, matchesPts[1]);                        
                    points3DColor.insert(points3DColor.end(), frameColor.begin(), frameColor.end());

                    pr->renderFrame(points3D, cameraLocations, points3DColor);                                                
                }                                    
                oldP = cameraProj.clone();
            }
                
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
        counter += 1;
    }    

    delete slam;
    delete matcher;    
    cap.release();
    cv::destroyAllWindows();

    while (!pr->quitWindow()) {
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
        pr->renderFrame(points3D, std::vector<cv::Mat>({}), points3DColor);
    }

    delete pr;
        
  
  
    return 0;
}
