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

*/

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <GL/glew.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#define VIDEO_PATH "../src/190416_10_Drone1_01.mp4"

void orbExtraction(cv::Ptr<cv::ORB> orb, cv::Mat frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    orb->detect(frame, keypoints);
    orb->compute(frame, keypoints, descriptors);   
}


int main()
{
    // test opencv 
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    
    std::vector<cv::KeyPoint> keyPoints = {};
    cv::Mat desc = cv::Mat();
    // matcher 
    cv::DescriptorMatcher *matcher = new cv::BFMatcher(cv::NormTypes::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches = {};

    //cv::VideoCapture cap(VIDEO_PATH);
    cv::VideoCapture cap(0);
    cv::Mat frame;
   
    while (cap.isOpened()) {                
        cap >> frame;
        cv::resize(frame, frame, cv::Size(720, 360));
        if (!frame.empty()) {       
            // front end
            cv::Mat oldDesc = desc.clone();
            std::vector<cv::KeyPoint> oldKp = keyPoints;
            
            orbExtraction(orb, frame, keyPoints, desc);
            matcher->match(oldDesc, desc, matches);

            std::sort(matches.begin(), matches.end());
            if (matches.size() >= 100) {
                matches.resize(100);
            }          

            cv::drawKeypoints(frame, keyPoints, frame, cv::Scalar(0, 255, 0));           
            for (int i = 0; i < matches.size(); i++) {
                cv::line(frame,
                    oldKp[matches[i].queryIdx].pt,
                    keyPoints[matches[i].trainIdx].pt,
                    cv::Scalar(255, 0, 0), 3);
            }
                
            
            cv::imshow("Frame", frame);
            char c = (char)cv::waitKey(25);
            if (c == 27)
                break;
        }else {
            break;
        }
    }

    delete matcher;
    cap.release();
    cv::destroyAllWindows();


    //pangolin::CreateWindowAndBind("Main", 640, 480);
    //glEnable(GL_DEPTH_TEST);

    //// Define Projection and initial ModelView matrix
    //pangolin::OpenGlRenderState s_cam(
    //    pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
    //    pangolin::ModelViewLookAt(0, 0, 1, 0, 0, 0, pangolin::AxisY)
    //);

    //// Create Interactive View in window
    //pangolin::Handler3D handler(s_cam);
    //pangolin::View& d_cam = pangolin::CreateDisplay()
    //    .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
    //    .SetHandler(&handler);

    //while (!pangolin::ShouldQuit())
    //{
    //    // Clear screen and activate view to render into
    //    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); /// wireframe mode
    //    d_cam.Activate(s_cam);

    //    // Render OpenGL Cube        
    //    pangolin::glDrawRect(-0.5f, -0.5f, 0.5f, 0.5);        

    //    // Swap frames and Process Events
    //    pangolin::FinishFrame();
    //}

    return 0;
}
