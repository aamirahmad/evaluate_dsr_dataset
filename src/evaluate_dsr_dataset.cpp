#include "evaluate_dsr_dataset.h"

void ImageOverlayer::initializeCamParams()
{
  int nframes=1;
  int n = 10; //Number of points used to process
  int N=nframes*n;
  
  vector<CvPoint2D32f> temp(n);
  vector<int> npoints;
  vector<CvPoint3D32f> objectPoints;
  vector<CvPoint2D32f> points[2];
  points[0].resize(n);
  points[1].resize(n);
  
  double R[3][3], T[3], E[3][3], F[3][3];
  double Q[4][4];
    

/*************************************************************/    
    
   _M1 = cvMat(3, 3, CV_64F, M1 );
   _M2 = cvMat(3, 3, CV_64F, M2 );
   _D1 = cvMat(1, 5, CV_64F, D1 );
   _D2 = cvMat(1, 5, CV_64F, D2 );
  CvMat _R = cvMat(3, 3, CV_64F, R );
  CvMat _T = cvMat(3, 1, CV_64F, T );
  CvMat _E = cvMat(3, 3, CV_64F, E );
  CvMat _F = cvMat(3, 3, CV_64F, F );
  CvMat _Q = cvMat(4,4, CV_64F, Q);
  
  vector<CvPoint2D32f>& pts = points[0];

  
  points[0][0].x=	344.0;
  points[0][0].y=	178.0;
	
  points[0][1].x=	693.0;
  points[0][1].y=	140.0;
	
  points[0][2].x=	1268.0;
  points[0][2].y=	415.0;
	
  points[0][3].x=	871.0;
  points[0][3].y=	177.0;
	
  points[0][4].x=	1108.0;
  points[0][4].y=	281.0;
	
  points[0][5].x=	437.0;
  points[0][5].y=	190.0;
	
  points[0][6].x=	786.0;
  points[0][6].y=	477.0;
	
  points[0][7].x=	363.0;
  points[0][7].y=	235.0;
	
  points[0][8].x=	527.0;
  points[0][8].y=	426.0;
  
  points[0][9].x=	761.0;
  points[0][9].y=	613.0;  

  
  
  points[1][0].x=	600.0;
  points[1][0].y=	743.0;
	
  points[1][1].x=	64.0;
  points[1][1].y=	522.0;
	
  points[1][2].x=	655.0;
  points[1][2].y=	247.0;
	
  points[1][3].x=	227.0;
  points[1][3].y=	389.0;
	
  points[1][4].x=	470.0;
  points[1][4].y=	287.0;
	
  points[1][5].x=	574.0;
  points[1][5].y=	594.0;
	
  points[1][6].x=	922.0;
  points[1][6].y=	294.0;
	
  points[1][7].x=	848.0;
  points[1][7].y=	540.0;
	
  points[1][8].x=	1001.0;
  points[1][8].y=	339.0;
  
  points[1][9].x=	1015.0;
  points[1][9].y=	282.0;  
  
  
  
  npoints.resize(nframes,n);
  objectPoints.resize(nframes*n);
  
  objectPoints[0] = cvPoint3D32f(5.0,-3.5,0.0);
  objectPoints[1] = cvPoint3D32f(1.0,-3.5,0.0);
  objectPoints[2] = cvPoint3D32f(1.0,+3.5,0.0);
  objectPoints[3] = cvPoint3D32f(0.0,-1.5,0.0);
  objectPoints[4] = cvPoint3D32f(0.0,+1.5,0.0);
  objectPoints[5] = cvPoint3D32f(4.3,-2.7,0.0);
  objectPoints[6] = cvPoint3D32f(4.3,+2.7,0.0);
  objectPoints[7] = cvPoint3D32f(5.25,-1.75,0.0);
  objectPoints[8] = cvPoint3D32f(5.25,+1.75,0.0);
  objectPoints[9] = cvPoint3D32f(5.0,+3.5,0.0); 
  
  for( int i = 1; i < nframes; i++ )
      copy( objectPoints.begin(), objectPoints.begin() + n,
      objectPoints.begin() + i*n );
  
  CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
  CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
  CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
  CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
  
   
   /**************************************************************************/
  double R1[3][3], R2[3][3], P1[3][4], P2[3][4];

  CvMat _R1 = cvMat(3, 3, CV_64F, R1);
  CvMat _R2 = cvMat(3, 3, CV_64F, R2);
  CvMat _P1 = cvMat(3, 4, CV_64F, P1);
  CvMat _P2 = cvMat(3, 4, CV_64F, P2);

/************************************** StereoCalibration - returns R T R1 R2 T1 T2 **************************************/   
  CvSize imageSize = cvSize(1294,964);

  cvStereoCalibrate( &_objectPoints, &_imagePoints1,
      &_imagePoints2, &_npoints,
      &_M1, &_D1, &_M2, &_D2,
      imageSize, &_R, &_T, &_E, &_F,
      cvTermCriteria(CV_TERMCRIT_ITER+
      CV_TERMCRIT_EPS, 30, 1e-5),
      CV_CALIB_USE_INTRINSIC_GUESS+ CV_CALIB_FIX_ASPECT_RATIO);

     
/*************************************************************************************************************************/ 

/***************************************** Extrinsic Parameters **********************************************************/
  CvMat* Rvec_left = cvCreateMat( 3, 1, CV_64F );
  Tvec_left = cvCreateMat( 3, 1, CV_64F );
  CvMat* Rvec_right = cvCreateMat( 3, 1, CV_64F );
  Tvec_right = cvCreateMat( 3, 1, CV_64F ); 
  
  cvFindExtrinsicCameraParams2(&_objectPoints, &_imagePoints1,&_M1,&_D1,Rvec_left,Tvec_left);
  
  Rvec_left_n = cvCreateMat( 3, 3, CV_64F );
  cvRodrigues2(Rvec_left,Rvec_left_n,0);
    

  cvFindExtrinsicCameraParams2(&_objectPoints, &_imagePoints2,&_M2,&_D2,Rvec_right,Tvec_right);
  
  Rvec_right_n = cvCreateMat( 3, 3, CV_64F );
  cvRodrigues2(Rvec_right,Rvec_right_n,0);
    
/*************************************************************************************************************************/
}


void ImageOverlayer::OverlayOnRightCam(double x, double y, double z, double thetaRob, CvScalar color, IplImage* baseImage, int sequenceNo, bool isOrientation)
{
   vector<CvPoint3D32f> Robot_PositionsToReProject;
   Robot_PositionsToReProject.resize(totPntsPerPosGT);
   
   CvMat _Robot_PositionsToReProject = cvMat(1, totPntsPerPosGT, CV_32FC3, &Robot_PositionsToReProject[0]);    
   
   Robot_PositionsToReProject[0] = cvPoint3D32f(x,y,z);
    for(int pts = 0; pts < circlePointsPerPosition; pts++) //circlePointsPerPosition points out of totPntsPerPosGT for circle
    {
      float theta = -M_PI + (float)pts*2*M_PI/(circlePointsPerPosition);  
      Robot_PositionsToReProject[1 + pts] = cvPoint3D32f( x + robRadius*cosf(theta), y + robRadius*sinf(theta), z);
    }
    
    if(isOrientation)
      for(int pts = 0; pts < arrowPointsPerPosition; pts++) //arrowPointsPerPosition points out of totPntsPerPos for th arrow
      {
	//color = color_est[0];
	Robot_PositionsToReProject[1 + circlePointsPerPosition + pts] = cvPoint3D32f( x + (float)pts*(robRadius/(float)arrowPointsPerPosition)*cosf(thetaRob), y + (float)pts*(robRadius/(float)arrowPointsPerPosition)*sinf(thetaRob), z);
      }    
    
    //orientation GT is not available for now
    vector<CvPoint2D32f> reprojectedPoints_Robot;
    reprojectedPoints_Robot.resize(totPntsPerPosGT);
    
    CvMat _imageReprojectedPoints_RobotRight = cvMat(1, totPntsPerPosGT, CV_32FC2, &reprojectedPoints_Robot[0]);
    
    cvProjectPoints2(&_Robot_PositionsToReProject, Rvec_right_n, Tvec_right, &_M2, &_D2, &_imageReprojectedPoints_RobotRight, NULL, NULL, NULL, NULL, NULL); 
   
    
    for(int pts = 0; pts < totPntsPerPosGT-arrowPointsPerPosition; pts++)
      {    
	CvPoint robot_PointToBeShownRight = cvPoint(reprojectedPoints_Robot[pts].x,reprojectedPoints_Robot[pts].y);
	cvCircle(baseImage, robot_PointToBeShownRight, 0, color, 2, 8, 0);  
	
      }
      
    if(isOrientation)  
    for(int pts = 0; pts < arrowPointsPerPosition; pts++)
      {    
	CvPoint robot_PointToBeShownRight = cvPoint(reprojectedPoints_Robot[1 + circlePointsPerPosition + pts].x,reprojectedPoints_Robot[1 + circlePointsPerPosition + pts].y);
	cvCircle(baseImage, robot_PointToBeShownRight, 0, color_est[0], 5, 8, 0);  
	
      }      
      
   //Put the sequenceNo on top of the image;
   CvFont fontLegend;
   double hScaleLegend=1.0;
   double vScaleLegend=1.0;
   int    lineWidthLegend=2;   
   cvInitFont(&fontLegend,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_NORMAL, hScaleLegend,vScaleLegend,2,lineWidthLegend);   
   char seqNo[50];
   sprintf(seqNo,"Image= %d",sequenceNo);
   //cvPutText(baseImage,seqNo, cvPoint(20,40), &fontLegend, cvScalar(24.0, 30.0, 56.0));
      
}


void ImageOverlayer::OverlayOnLeftCam(double x, double y, double z, double thetaRob, CvScalar color, IplImage* baseImage, int sequenceNo, bool isOrientation)
{
   vector<CvPoint3D32f> Robot_PositionsToReProject;
   Robot_PositionsToReProject.resize(totPntsPerPosGT);
   
   CvMat _Robot_PositionsToReProject = cvMat(1, totPntsPerPosGT, CV_32FC3, &Robot_PositionsToReProject[0]);    
   
   Robot_PositionsToReProject[0] = cvPoint3D32f(x,y,z);
    for(int pts = 0; pts < circlePointsPerPosition; pts++) //circlePointsPerPosition points out of totPntsPerPosGT for circle
    {
      float theta = -M_PI + (float)pts*2*M_PI/(circlePointsPerPosition);  
      Robot_PositionsToReProject[1 + pts] = cvPoint3D32f( x + robRadius*cosf(theta), y + robRadius*sinf(theta), z);
    }
    
    if(isOrientation)
      for(int pts = 0; pts < arrowPointsPerPosition; pts++) //arrowPointsPerPosition points out of totPntsPerPos for th arrow
      {
	//color = color_est[0];
	Robot_PositionsToReProject[1 + circlePointsPerPosition + pts] = cvPoint3D32f( x + (float)pts*(robRadius/(float)arrowPointsPerPosition)*cosf(thetaRob), y + (float)pts*(robRadius/(float)arrowPointsPerPosition)*sinf(thetaRob), z);
      }    
    
    //orientation GT is not available for now
    vector<CvPoint2D32f> reprojectedPoints_Robot;
    reprojectedPoints_Robot.resize(totPntsPerPosGT);
    
    CvMat _imageReprojectedPoints_RobotRight = cvMat(1, totPntsPerPosGT, CV_32FC2, &reprojectedPoints_Robot[0]);
    
    cvProjectPoints2(&_Robot_PositionsToReProject, Rvec_left_n, Tvec_left, &_M1, &_D1, &_imageReprojectedPoints_RobotRight, NULL, NULL, NULL, NULL, NULL); 
   
    
    for(int pts = 0; pts < totPntsPerPosGT-arrowPointsPerPosition; pts++)
      {    
	CvPoint robot_PointToBeShownRight = cvPoint(reprojectedPoints_Robot[pts].x,reprojectedPoints_Robot[pts].y);
	cvCircle(baseImage, robot_PointToBeShownRight, 0, color, 2, 8, 0);  
	
      }
      
    if(isOrientation)  
    for(int pts = 0; pts < arrowPointsPerPosition; pts++)
      {    
	CvPoint robot_PointToBeShownRight = cvPoint(reprojectedPoints_Robot[1 + circlePointsPerPosition + pts].x,reprojectedPoints_Robot[1 + circlePointsPerPosition + pts].y);
	cvCircle(baseImage, robot_PointToBeShownRight, 0, color_est[0], 5, 8, 0);  
	
      }      
      
   //Put the sequenceNo on top of the image;
   CvFont fontLegend;
   double hScaleLegend=1.0;
   double vScaleLegend=1.0;
   int    lineWidthLegend=2;   
   cvInitFont(&fontLegend,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_NORMAL, hScaleLegend,vScaleLegend,2,lineWidthLegend);   
   char seqNo[50];
   sprintf(seqNo,"Image= %d",sequenceNo);
   cvPutText(baseImage,seqNo, cvPoint(20,40), &fontLegend, cvScalar(24.0, 30.0, 56.0));
      
}


void ImageOverlayer::gtDataCallback(const read_dsr_dataset::MBOTpersonGTData::ConstPtr& msg, IplImage* img_)
{
  // The filenames below are important to locate the exact stereo camera images which were used to generate the GT of person and the robot.
  string imageFileRight = msg->RightFilename; 
  string imageFileLeft = msg->LeftFilename;
  
  //actual message data stored in local variables
  foundMBOT_GT = msg->foundMBOT;
  foundPerson_GT = msg->foundperson;
  mbotGTPose = msg->poseMBOT;
  mbotPersonGTPose = msg->posePerson;
  
  
  //we do all overlaying on the right camera images. feel free to do it similarly for left camera.
  rightcamActive = msg->rightcamactive;
  leftcamActive = msg->leftcamactive;
  //forcing right camera to be active
  rightcamActive = 1;  leftcamActive = 0;

  
  int datasetNumber;
  char imagePathName[500];
  strcpy(imagePathName,imagePathNameBaseName);  
  char imageFilename[100];
  sscanf(imageFileRight.c_str(),"/store/DataSet_Lisbon/PersonTrackingDataset/GT_%d/Right/%s",&datasetNumber,imageFilename);  // This is a small hack to extract the exact file name of the image from the rosbag message. The whole directory path was saved at the time of bag creation. This is simply a safe workaround. 
  strcat(imagePathName,imageFilename);
  //cout<<"filename = "<<imagePathName<<endl;
  
  img_=cvLoadImage(imagePathName);
  
  double actualRobTheta;  

    if(foundMBOT_GT) // a good GT estimate exists
    {    
      double tempPosX = mbotGTPose.pose.position.x; 
      double tempPosY = mbotGTPose.pose.position.y; 
      double tempPosZ = mbotGTPose.pose.position.z; 
      
      tf::Quaternion quat;
      tf::quaternionMsgToTF(mbotGTPose.pose.orientation, quat);
      tf::Matrix3x3 m(quat);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      actualRobTheta = yaw;
      if (actualRobTheta >  M_PI)
	actualRobTheta = -M_PI + (actualRobTheta-M_PI);
      if (actualRobTheta < -M_PI)
      actualRobTheta =  M_PI + (actualRobTheta+M_PI);      
    
      if(rightcamActive==1)
      {
	OverlayOnRightCam(tempPosX,tempPosY,tempPosZ,actualRobTheta,color_est[4],img_,msg->header.seq,true);
	
	for(int i=0; i<=30;i++)
	  OverlayOnRightCam(tempPosX,tempPosY,(tempPosZ*i)/30,actualRobTheta,color_est[4],img_,msg->header.seq,false); 
      }
      if(leftcamActive==1)
      {
	OverlayOnLeftCam(tempPosX,tempPosY,tempPosZ,actualRobTheta,color_est[4],img_,msg->header.seq,true);
	
	for(int i=0; i<=30;i++)
	  OverlayOnLeftCam(tempPosX,tempPosY,(tempPosZ*i)/30,actualRobTheta,color_est[4],img_,msg->header.seq,false); 
      }	
	
    }
    
    if(foundMBOT_GT && foundPerson_GT)
    {
      double tempPosX = mbotPersonGTPose.pose.position.x; 
      double tempPosY = mbotPersonGTPose.pose.position.y; 
      double tempPosZ = mbotPersonGTPose.pose.position.z; 
      
      if(mbotPersonGTPose.pose.position.x>8 && mbotPersonGTPose.pose.position.y<-4) // these values correspond to 0,0 in pixels and hence discard these gt! (happens before the person enters the scene)
      {
	foundPerson_GT = 0;
	//cout<<"bad GT"<<endl;
      }
      
      double tempPosTheta = 0; // person's orientation is not considered
    
      //same overlayer as robot gt can be used with a taller bounding cylinder
      if(rightcamActive==1 && foundPerson_GT)
      {      
	for(int i=0; i<=50;i++)
	  OverlayOnRightCam(tempPosX,tempPosY,(1.5*i)/50,tempPosTheta,color_est[5],img_,msg->header.seq,false);     
	gt_successCounter++;
      }
      if(leftcamActive==1 && foundPerson_GT)
      {
	for(int i=0; i<=50;i++)
	  OverlayOnLeftCam(tempPosX,tempPosY,(1.5*i)/50,tempPosTheta,color_est[5],img_,msg->header.seq,false);  
	gt_successCounter++;	
      }

      
    }

    if(foundMBOT_GT && foundPerson_GT && foundPerson)
    {

      //Convert person's pose estimated by the robot to the global frame before performing overlaying

      //storing in local variables the person's estimated poses by the mbot in the mbot's reference frame
      double tempPosTheta = 0;
      double tempPosX = mbotPersonPose.pose.position.x;
      double tempPosY = mbotPersonPose.pose.position.y;
      double tempPosZ = mbotPersonPose.pose.position.z;
      
      //now converting the above robot-frame position to the world frame (remember that world frame coincides here with the frame of the GT so we simply use the robot's GT pose to perform this conversion)
      double worldPerX,worldPerY,worldPerZ;
      
      worldPerX = mbotGTPose.pose.position.x + tempPosX*cos(actualRobTheta) - tempPosY*sin(actualRobTheta);
      worldPerY = mbotGTPose.pose.position.y + tempPosX*sin(actualRobTheta) + tempPosY*cos(actualRobTheta);
      worldPerZ = mbotPersonGTPose.pose.position.z + tempPosZ;
      
      
      double dist_gt_estX,dist_gt_estY;
      dist_gt_estX = fabs(mbotPersonGTPose.pose.position.x - worldPerX);
      dist_gt_estY = fabs(mbotPersonGTPose.pose.position.y - worldPerY);
      double err_gt_dist=0.0;
      err_gt_dist = pow((dist_gt_estX*dist_gt_estX + dist_gt_estY*dist_gt_estY),0.5);
      //cout<<" error in the person position = "<<err_gt_dist<<endl;
      
      //same overlayer as robot gt can be use
      if(rightcamActive==1 /*&& err_gt_dist <1.0*/)
      {      
	for(int i=0; i<=50;i++)
	  OverlayOnRightCam(worldPerX,worldPerY,(1.5*i)/50,tempPosTheta,color_est[2],img_,msg->header.seq,false);     
	successfulDetections++;
      }
      if(leftcamActive==1 /*&& err_gt_dist <1.0*/)
      {
	for(int i=0; i<=50;i++)
	  OverlayOnLeftCam(worldPerX,worldPerY,(1.5*i)/50,tempPosTheta,color_est[2],img_,msg->header.seq,false);   
        successfulDetections++;
      }      
      

      std_msgs::Float32 error_msg;
      error_msg.data = err_gt_dist;
      errorPerson_inMBOTref.publish(error_msg);
      
      errX[residualCounter] = dist_gt_estX;
      errY[residualCounter] = dist_gt_estY;
    
      finalRMSmean = finalRMSmean + err_gt_dist;
      residualCounter++;

            
      ///Doing the error analysis here once we have reached the end of the dataset.
      double endPointCounter=0;
      if(datasetNumber == 2) // for Dataset_1
	endPointCounter = 5600;
      if(datasetNumber == 3) // for Dataset_2
	endPointCounter = 3997;      
      
      if(msg->header.seq==endPointCounter && endPointCounter!=0)
      {
	finalRMSmean = finalRMSmean/residualCounter;
	
	  for(int i=0;i<=residualCounter;i++)
	    {
		  double rms = pow(errX[i]*errX[i] + errY[i]*errY[i],0.5);
		  finalRMSvar = finalRMSvar + pow((rms-finalRMSmean),2);
	    }
	    finalRMSvar = finalRMSvar/residualCounter;
	    finalRMSdev = pow(finalRMSvar,0.5);
	    cout<<"Person pose average error, RMSD and RMS w.r.t. GT = "<<finalRMSmean<<"  "<<finalRMSdev<<"  "<<finalRMSvar<<endl;
	    cout<<"successfulDetections = "<<successfulDetections<<endl;
	    cout<<"gt_successCounter = "<<gt_successCounter<<endl<<endl;	
      }
    }
    
  cvShowImage("Grount Truth Images: Right Camera", img_);
  cvReleaseImage(&img_);

}


void ImageOverlayer::mbotCallback(const read_dsr_dataset::MBOTpersonGTData::ConstPtr& msg, IplImage* img_)
{
  foundPerson = msg->foundperson; // your estimator found the person
  mbotPersonPose = msg->posePerson; //this is pose of the person in the robot's local frame that is found by your estimator
  foundMBOT = msg->foundMBOT;
  mbotPose = msg->poseMBOT;
  //overlaying is not performed in this method. see the gtcallback for the overlaying and error calculation
}

// void ImageOverlayer::mbotAMCLCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg, IplImage* img_)
// {
// 
//   mbotAMCLPose = *msg;
//   
//       tf::Quaternion quat;
//       tf::quaternionMsgToTF(mbotAMCLPose.pose.pose.orientation, quat);
//       tf::Matrix3x3 m(quat);
//       double roll, pitch, yaw;
//       m.getRPY(roll, pitch, yaw);
//       robAMCLTheta = yaw;
//       
//   //cout<<" robot orientation from AMCL is = "<<robAMCLTheta<<endl;
//   //cout<<" robot orientation from GT is = "<<robGTTheta<<endl;
//   //cout<<" at time  = "<<mbotAMCLPose.header.stamp <<" robot orientation diff in AMCL and GT is = "<<robGTTheta-robAMCLTheta<<endl;
// }

int main(int argc, char **argv)
{
  ros::init(argc, argv, "evaluate_overlay_dsr_dataset"); 
  
    if (argc != 2)
         {
            ROS_WARN("WARNING: you should specify images folder path (on which the GT and esimates are overlaid)! which might be %s\n",argv[1]);
         }else{
             printf("INFO: you have set images folder path: %s\n",argv[1]);
         }  

  cvStartWindowThread();

  ImageOverlayer node(argv[1]);
  
  ros::spin();
  return 0;
  
}
