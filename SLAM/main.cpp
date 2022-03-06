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
*/

#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <GL/glew.h>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>

#include <Eigen/StdVector>
#include <opencv2/core/eigen.hpp>

#include <pangolin/gl/gl.h>



#define VIDEO_PATH "../src/190416_10_Drone1_01.mp4"
//#define VIDEO_PATH 0


void orbExtraction(cv::Ptr<cv::ORB> orb, cv::Mat frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {    
    orb->detect(frame, keypoints);
    orb->compute(frame, keypoints, descriptors);   
}

class PangolinRenderer {
public:    
    PangolinRenderer() {   
        pangolin::CreateWindowAndBind("Main", 640, 480);
        glEnable(GL_DEPTH_TEST);

        s_cam = new pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin::ModelViewLookAt(0, 0, 1, 0, 0, 0, pangolin::AxisY)
        );

        handler  = new pangolin::Handler3D(*s_cam);       
        d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(handler);
    }

    ~PangolinRenderer() {
        delete s_cam;
        delete handler;
    }

    void renderFrame(std::vector<cv::Point2f> cvPositions, std::vector<float> cols) {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); /// wireframe mode
        d_cam.Activate(*s_cam);                
        
        this->draw(cvPositions, cols);

        // Swap frames and Process Events
        pangolin::FinishFrame();       
    }       

    void draw(std::vector<cv::Point2f> cvPositions, std::vector<float> cols) {
        this->draw3DPoints(cvPositions, cols);  
        this->envTools();
    }

    void envTools(){
        pangolin::glDrawAxis(0.2f);
    }

    void draw3DPoints(std::vector<cv::Point2f> cvPositions, std::vector<float> cols) {
        std::vector<float> points = {};
        for (cv::Point2f cvp : cvPositions) {
            points.push_back(cvp.x/1000);
            points.push_back(cvp.y/1000);
            points.push_back(0.f);            
        }        
        pangolin::glDrawColoredVertices<float, float>((int)points.size() / 3, points.data(), cols.data(), GL_POINTS, 3, 3);
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

    std::vector<float> getColors(cv::Mat frame, std::vector<cv::Point2f> pts) {
        std::vector<float> colors = {};
        // extract informations
        for (cv::Point2f point : pts) {
            //add colors  
            colors.push_back(frame.at<cv::Vec3b>(point)[0]);
            colors.push_back(frame.at<cv::Vec3b>(point)[1]);
            colors.push_back(frame.at<cv::Vec3b>(point)[2]);
        }
        return colors;
    }
    
    std::vector<std::vector<cv::Point2f>> getMatchesPoints() {
        std::vector<std::vector<cv::Point2f>> matchPoints = { {}, {} };

        // extract informations
        for (int i = 0; i < matches.size(); i++) {   
            cv::Point2f p1;
            cv::Point2f p2;
            p1 = this->oldKps[std::min((int)matches[i].queryIdx, (int)this->oldKps.size() - 1)].pt;
            p2 = this->newKps[std::min((int)matches[i].trainIdx, (int)this->newKps.size() - 1)].pt;
            matchPoints[0].push_back(p1);
            matchPoints[1].push_back(p2);
        }
        return matchPoints;
    }

    std::vector<cv::KeyPoint> getKps() {
        return this->newKps;
    }

    void match(cv::Mat frame) {
        this->matcher->match(this->oldDescrs, this->newDescrs, this->matches);
        std::sort(this->matches.begin(), this->matches.end());

        if (this->matches.size() >= 100) {
            this->matches.resize(this->maxNMatches);
        }        
    }

    
    void findCameraMatrices() {                   
        std::vector<std::vector<cv::Point2f>> matchesPts = this->getMatchesPoints();
        // find homography and fundamental matrix in "parallel"
        cv::Mat H = cv::findHomography(matchesPts[0], matchesPts[1]);
        cv::Mat F = cv::findFundamentalMat(matchesPts[0], matchesPts[1]);

        std::cout << H << std::endl;
        std::cout << F << std::endl;
    }

    std::vector<std::vector<cv::Point2f>> matchesPts = {};

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
    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1); // instantiate LSH index parameters
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);       // instantiate flann search parameters
    cv::DescriptorMatcher* matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
    Slammer* slam = new Slammer(orb, matcher);
    
    // video aquisition 
    cv::VideoCapture cap(VIDEO_PATH);
    cv::Mat frame;
   
    // pangolin settings
    PangolinRenderer* pr = new PangolinRenderer();

    int counter = 0;
    while (cap.isOpened()) {                    
        cap >> frame;
        cv::resize(frame, frame, cv::Size(720, 360));
        if (counter % 10 == 0) {
            if (!frame.empty()) {
                slam->detect(frame);           
                // front end
                // render pangolin window                
                std::vector<std::vector<cv::Point2f>> matchesPts = slam->getMatchesPoints();
                pr->renderFrame(matchesPts[1], slam->getColors(frame, matchesPts[1]));
                            
                // triangulate points
                cv::drawKeypoints(frame, slam->getKps(), frame, cv::Scalar(0, 255, 0));

                for (int i = 0; i < matchesPts[0].size(); i++) {
                    cv::line(frame,
                        matchesPts[0][i],
                        matchesPts[1][i],
                        cv::Scalar(255, 0, 0), 2);
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
