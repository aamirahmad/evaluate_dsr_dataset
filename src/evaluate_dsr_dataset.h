#include <cstdio>
#include <iostream>
#include <string>

#include "ros/ros.h"
#include <rosbag/bag.h>
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"

#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>
#include <read_dsr_dataset/MBOTpersonGTData.h>


using namespace cv;
using namespace std;
using namespace ros;


const std::size_t NUM_ROBOTS = 1; // 1 mbot
const std::size_t NUM_TARGETS = 1; // 1 person
const std::size_t MAX_IMAGES = 6000; // 1 person
//Right Camera PARAMETERS (CALIBRATED FROM BEFORE)
/*** Intrinsic Parameters********/
  // LRM new GT system (ethernet)
  //Left camera
  double M1[] = {952.364130338909490,0,638.552089454614702, 0,953.190083002521192,478.055570057081638, 0,0,1};
  double D1[] = {-0.239034777360406 , 0.099854524375872 , -0.000493227284393 , 0.000055426659564 , 0.000000000000000 };  
  
  //Right Camera
  double M2[] = {949.936867662258351,0,635.037661574667936,0,951.155555785186380,482.586148439616579 ,0,0,1};
  double D2[] = { -0.243312251484708 , 0.107447328223015 , -0.000324393194744 , -0.000372030056928 , 0.000000000000000 };

  //static variables used for overly design stuff!!
  static const int circlePointsPerPosition = 50;
  static const int targetPntsPos = 1;
  static const int arrowPointsPerPosition = 20;
  static const int totPntsPerPos = 1 + circlePointsPerPosition + arrowPointsPerPosition;
  static const int totPntsPerPosGT = 1 + circlePointsPerPosition + arrowPointsPerPosition;
  static const float robRadius = 0.30; //in meters
  
  
  
class ImageOverlayer
{
  IplImage img; 
  
  char imagePathNameBaseName[500];
  
  NodeHandle n; 
  Subscriber mbot_person_GTsub, mbot_estimated_sub, mbot_amcl_sub;
  
  Publisher errorPerson_inMBOTref;
  
  CvMat _M2;
  CvMat _D2;
  CvMat _M1;
  CvMat _D1;
  CvMat* Tvec_right;
  CvMat* Rvec_right_n;  
  CvMat* Tvec_left;
  CvMat* Rvec_left_n;  
  CvScalar color_est[6];
  
  //Robot and target's GT poses and positions
  bool foundMBOT_GT;
  bool foundPerson_GT;
  geometry_msgs::PoseWithCovariance mbotGTPose;
  geometry_msgs::PoseWithCovariance mbotPersonGTPose;
  int rightcamActive;
  int leftcamActive;
  int whichCamAcive; //1 for right, 2 for left, just to know which of the two GT cameras is active.
  
  //Estimatd robot and target states in the WORLD FRAME (World frame coincides with the GT system's frame of reference)
  bool foundMBOT;
  bool foundPerson;
  geometry_msgs::PoseWithCovariance mbotPose; //either comes from another method of estimation or simply use the existing amcl from the dataset or the ground truth.  
  geometry_msgs::PoseWithCovariance mbotPersonPose; 
  
  // Mbot dataset already has an amcl-generated pose of the robot. Though this is not in the right world frame, transformations to the correct one are provided in the implementation of this class.
  //   geometry_msgs::PoseWithCovarianceStamped mbotAMCLPose;
  //   double robAMCLTheta;
  //   double robGTTheta;
  
  //some counters and auxilliary variables.
  int successfulDetections;
  int gt_successCounter;
  
  
  int errorCounter;
  int maxVertexCount;
  double finalRMSmean ;
  double finalRMSvar;
  double finalRMSdev;
  int residualCounter;
  double errX[MAX_IMAGES];
  double errY[MAX_IMAGES];
  
  public:
    
    ImageOverlayer(char *camImageFolderPath)
    {
     strcpy(imagePathNameBaseName,camImageFolderPath); 
      
     cvNamedWindow("Grount Truth Images: Right Camera"); 
     
     mbot_person_GTsub = n.subscribe<read_dsr_dataset::MBOTpersonGTData>("mbot_person_gt_data", 1000, boost::bind(&ImageOverlayer::gtDataCallback,this,_1,&img)); 

     
     mbot_estimated_sub = n.subscribe<read_dsr_dataset::MBOTpersonGTData>("mbot_person_estimated_data", 1000, boost::bind(&ImageOverlayer::mbotCallback,this,_1,&img));  
     
     errorPerson_inMBOTref = n.advertise<std_msgs::Float32>("errorPerson_inMBOTref", 1000);
     
     initializeCamParams();
     
     
     //change these to alter colors of the robots estimated state representation
     color_est[0] = cvScalar(0.0, 0.0, 255.0);
     color_est[1] = cvScalar(0.0, 0.0, 0.0);
     color_est[2] = cvScalar(20.0, 171.0, 20.0);
     color_est[3] = cvScalar(147.0, 20.0, 255.0);
     color_est[4] = cvScalar(255.0, 50.0, 0.0);
     color_est[5] = cvScalar(100.0, 100.2, 250.1);
     
     foundMBOT_GT = false;
     successfulDetections = 0;
     gt_successCounter=0;
     
     errorCounter = 0;
     maxVertexCount = 6000;
     finalRMSmean = 0.0;
     finalRMSvar = 0.0;
     finalRMSdev = 0.0;
     residualCounter = 0;
    }
    
    
    ///calbback that calls the methods OverlayEstimatedRobotPose and OverlayGTRobotPose whenever a new GT message is received from the bag
    void gtDataCallback(const read_dsr_dataset::MBOTpersonGTData::ConstPtr& , IplImage* );
    
    ///callback that keeps updating the most recent robot poses for mbot when it receives this info from the actual estimator (implemented seprately by the user), and assumed to be 0,0 in case no such estimator is running.
    void mbotCallback(const read_dsr_dataset::MBOTpersonGTData::ConstPtr& , IplImage* );
    
    
    ///Important initializer: computes the reprojection matrices based on the stereo calibration. Do not change this method for the LRM GT images!!! Port it to other stereosystems if and when necessary
    void initializeCamParams(void);
    
    ///Overlay the black circle for the updated GT pose of the robot
    void OverlayOnRightCam(double, double, double, double, CvScalar, IplImage*, int, bool);
    
    ///Overlay the black circle for the updated GT pose of the robot on the left stream
    void OverlayOnLeftCam(double, double, double, double, CvScalar, IplImage*, int, bool);    
    
};



