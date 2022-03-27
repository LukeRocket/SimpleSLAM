


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
//#define VIDEO_PATH "../src/190416_10_Drone1_01.mp4"
#define VIDEO_PATH "../src/test.mp4"

//#define FRAME_WIDTH 720
//#define FRAME_HEIGHT 360
#define FRAME_WIDTH (int)1920/2
#define FRAME_HEIGHT (int)1080/2
typedef uint8_t Pixel;

void orbExtraction(cv::Ptr<cv::ORB> orb, cv::Mat frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {                
    cv::Mat meanFrame = cv::Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
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
            pangolin::ProjectionMatrix(FRAME_WIDTH, FRAME_HEIGHT, 420, 420, FRAME_WIDTH/2.f, FRAME_HEIGHT/2.f, 0.2, 100000),
            pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY)
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

    void setCameraLocation(cv::Mat cameraCoords, float distanceFromKf = 10.f) {
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
            pangolin::ProjectionMatrix(FRAME_WIDTH, FRAME_HEIGHT, 420, 420, FRAME_WIDTH / 2.f, FRAME_HEIGHT / 2.f, 0.2, 10000),
            pangolin::ModelViewLookAt(
                //0,0,0,
                cameraCenterLocation[0], cameraCenterLocation[1]+distanceFromKf, cameraCenterLocation[2]+distanceFromKf,
                lookAtVertex[0], lookAtVertex[1], lookAtVertex[2], pangolin::AxisY)
        );


        this->handler = new pangolin::Handler3D(*this->s_cam);
        this->d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -(float)this->width / (float)this->height)
            .SetHandler(this->handler);
    }

    void renderFrame(cv::Mat cvPointsPositions, std::vector<cv::Mat> KeyFramesPositions, std::vector<float> cols3DPoints = {}) {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); /// wireframe mode

        if (KeyFramesPositions.empty() != true) {
            this->setCameraLocation(KeyFramesPositions[KeyFramesPositions.size() - 1], 15.f);
        }
        
        d_cam.Activate(*s_cam);                
                
        // draw points
        this->drawKeyFrames(KeyFramesPositions);
        this->draw3DPoints(cvPointsPositions, cols3DPoints);
        
        // Swap frames and Process Events
        pangolin::glDrawAxis(0.2f);
        pangolin::FinishFrame();       
    }       
   
    void drawKeyFrames(std::vector<cv::Mat> positions) {
                                        
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

    void draw3DPoints(cv::Mat cvPositions, std::vector<float> cols, float squareDim = .1f ) {
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
                if (matchesList[i][0].distance < 0.65f * matchesList[i][1].distance){
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

    std::array<cv::Mat, 3> singularValueDecomp(cv::Mat matrix) {
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
        std::cout << svd.singularValues() << std::endl;
        
        std::cout << svd.singularValues()  << std::endl;
        std::cout << svd.matrixU()  << std::endl;
        std::cout << svd.matrixV()  << std::endl;

        cv::eigen2cv(svd.singularValues(), sigma);        
        cv::eigen2cv(svd.matrixU(), u);
        cv::eigen2cv(svd.matrixV(), v);
        
        std::array<cv::Mat, 3> output = {sigma, u, v};
        return output;
    }

    float estimateCameraFocalLength(cv::Mat F, std::vector<float> &focalLengths) {
        /*
            Estimate Focal length from successive estimations of the fundamental matrix
            (Kruppa equations)         
        */                     
        std::array<cv::Mat, 3> decomp = this->singularValueDecomp(F);
           
        

        return median(focalLengths);
    }

    void applyMaskToMatches(std::vector<cv::Mat> &matchesPts, cv::Mat mask) {

        cv::Mat out1 = {};
        cv::Mat out2 = {};

        for (int rIndex = 0; rIndex < mask.rows; rIndex++) {
            if (mask.at<uchar>(rIndex, 0) != 0) {
                out1.push_back(matchesPts[0].at<cv::Point2f>(rIndex, 0));
                out2.push_back(matchesPts[1].at<cv::Point2f>(rIndex, 0));
            }
        }
        
        /*matchesPts[0].copyTo(out1, mask);
        matchesPts[1].copyTo(out2, mask);
        */
        matchesPts[0] = out1.clone();
        matchesPts[1] = out2.clone();
    }

    cv::Mat getCameraPoseMatrix(cv::Mat camMatrix, std::vector<cv::Mat> &matchesPts) {        
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
        //cv::hconcat(R.t(), -R.t() * t, cameraPoseMatrix);
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
            endLocations.push_back((camPose * startLocations.row(rIndex).t()).t());
        }

        return endLocations;
    }   

    cv::Mat getCurrentCameraTransform(cv::Mat newPose, cv::Mat oldPose) {
        std::vector<float> lastRowData = { 0, 0, 0, 1 };
        cv::Mat lastRow = cv::Mat(1, 4, CV_32F, lastRowData.data());
        
        newPose.convertTo(newPose, CV_32F);
        cv::vconcat(newPose, lastRow, newPose);
        oldPose.convertTo(oldPose, CV_32F);
        cv::vconcat(oldPose, lastRow, oldPose);

        cv::Mat cameraTransform = newPose * oldPose;        
        return cameraTransform.rowRange(0, 3);
    }


    cv::Mat dehomogenizedCoords(cv::Mat points3DHomog) {
        cv::Mat point3dDehomog = {};
        
        assert(points3DHomog.cols == 4);

        for (int rIndex = 0; rIndex < points3DHomog.rows; rIndex ++) {
            cv::Mat row = points3DHomog.row(rIndex);
            for (int cIndex = 0; cIndex < 3; cIndex++)
                row.at<float>(0, cIndex) /= -row.at<float>(0, 3);
            row(cv::Range(0, 1), cv::Range(0, 3)).copyTo(row);
            point3dDehomog.push_back(row);
        }
        return point3dDehomog;
    }

    cv::Mat get3DPoints(std::vector<cv::Mat> matchesPts, cv::Mat oldProjectMatrix, cv::Mat projectMatrix) {
       
        cv::Mat points3D = {};       
        cv::triangulatePoints(oldProjectMatrix, projectMatrix, matchesPts[0].t(), matchesPts[1].t(), points3D);
       
        /*this->homogCoords(&matchesPts[1]);
        std::vector<cv::Mat*> matrices = { &projectMatrix, &matchesPts[1]};
        standardizeMatricesType(matrices);        
        
        cv::Mat points3D = projectMatrix.t() * matchesPts[1].t();        */
        return this->dehomogenizedCoords(points3D.t());                
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
    float focalLenght = 270.f;
    // define camera matrix
    cv::Mat camMatrix = cv::Mat::zeros(3, 3, CV_32F);

    // set camera center 
    camMatrix.at<float>(0, 2) += FRAME_WIDTH / 2;
    camMatrix.at<float>(1, 2) += FRAME_HEIGHT / 2;

    // set focal lenght
    camMatrix.at<float>(0, 0) += focalLenght;
    camMatrix.at<float>(1, 1) += focalLenght;
    camMatrix.at<float>(2, 2) += 1.f;   

    cv::Mat oldP = {};
    cv::Mat cameraProj = {};
    cv::Mat points3D = {};
    std::vector<float> points3DColor = {};

    // normalized w.r.t. frame width (considered equal to 1)    
    std::vector<float> locationData = { -0.5f, 0.f, 0.f, 1.f,
                                        -0.5f,  FRAME_HEIGHT / (float)(FRAME_WIDTH) , 0.f, 1.f,
                                         0.5f,  FRAME_HEIGHT / (float)(FRAME_WIDTH) , 0.f, 1.f,
                                         0.5f, 0.f, 0.f, 1.f,
                                         0.f, FRAME_HEIGHT / (float)(2 * FRAME_WIDTH), camMatrix.at<float>(0, 0)/(float)FRAME_WIDTH, 1.f}; // camera center point
    std::vector<cv::Mat> cameraLocations = { cv::Mat(5, 4, CV_32F, locationData.data()) };

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
                //slam->normalizeCoords(&matchesPts[0], camMatrix);
                //slam->normalizeCoords(&matchesPts[1], camMatrix);
                
                cv::Mat cameraPose = slam->getCameraPoseMatrix(camMatrix, matchesPts);
                cv::Mat newCamLocation = slam->getCameraLocation(cameraLocations[cameraLocations.size() - 1], cameraPose);
               
                if (oldCameraPose.empty() == false) {
                    cameraPose = slam->getCurrentCameraTransform(cameraPose, oldCameraPose);                        
                }
                oldCameraPose = cameraPose.clone();

                cameraLocations.push_back(newCamLocation);
                cameraProj = slam->getCameraProjectionMatrix(cameraPose, camMatrix);
                    
                if (oldP.empty() == false) {
                    /*slam->normalizeCoords(&matchesPts[0], camMatrix);
                    slam->normalizeCoords(&matchesPts[1], camMatrix);*/

                    cv::Mat pInFrame = slam->get3DPoints(matchesPts, oldP, cameraProj);
                    //cv::Mat pInFrame = slam->get3DPoints(matchesPts, oldP, cameraPose);

                    for (int rIndex = 0; rIndex < pInFrame.rows; rIndex++) {
                        points3D.push_back(pInFrame.row(rIndex));
                    }

                    /*slam->denormalizeCoords(&matchesPts[1], camMatrix);
                    slam->denormalizeCoords(&matchesPts[0], camMatrix);*/
                        
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

    while (true) {
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
        pr->renderFrame(points3D, std::vector<cv::Mat>({}), points3DColor);
    }

    delete pr;
        
  
  
    return 0;
}
