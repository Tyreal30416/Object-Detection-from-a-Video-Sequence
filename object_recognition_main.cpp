#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <algorithm>

#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/features2d/features2d.hpp"

#include "ldb.h"
#include "utility.h"

using namespace std;
using namespace cv;

// function for computing the homography between two images
int computeHomography(Mat& image_1, vector<KeyPoint>& keypoints_1, Mat& descriptors_1, 
							Mat& image_2, const int flag, const int minGoodMatches, Mat& resultedHomography) {

	Mat descriptors_2;
	vector<KeyPoint> keypoints_2;
	vector<Point2f> obj, scene; 
	
	ORB orb;
	orb(image_2, Mat(), keypoints_2, descriptors_2);
	LDB ldb(48);
	ldb.compute(image_2, keypoints_2, descriptors_2, flag);
	
	//-- Keypoint Matching
	DescriptorMatcher* pMatcher = new BFMatcher(NORM_HAMMING, false);//brute-force
	vector< vector<DMatch> > matches;//all matched pairs
	vector<DMatch> good_matches;//good matched pairs
	//for each key point, find two mached points: the best point and the better point
	if ((descriptors_1.type() == descriptors_2.type()) && 
		(descriptors_1.cols == descriptors_2.cols)) {
			pMatcher->knnMatch(descriptors_1, descriptors_2, matches, 2);
	} 
	else {
		return 0;
	}

	delete pMatcher;
	 
	for(unsigned int i=0; i<matches.size(); i++){
		//if the best point is ditinguished from the better one, keep it
	  if(matches[i][0].distance < 0.8*matches[i][1].distance){
	     good_matches.push_back(matches[i][0]);
	  }
	}
	
  //draw the good-macthed pairs
	Mat out_img;
	drawMatches(image_1, keypoints_1, image_2, keypoints_2, good_matches, out_img);
	imshow("Good matches between pattern image and warped frame", out_img);
	char key = waitKey(20);

	//-- RANSAC geometric verification
	if(good_matches.size() < minGoodMatches){
	     cout << "insufficient matches for RANSAC verification," << good_matches.size()  << endl;
	     return 0; //recognition failure
	} 
	for(unsigned int i = 0; i < good_matches.size(); i++) {  
	   obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );  
	   scene.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );   
	}  
		
	cout << "The number of good matched pairs is " << good_matches.size() << endl; 
	//current homography   
	resultedHomography = findHomography( obj, scene, CV_RANSAC );	
	return 1; //sucessful
}


/************** Main Program ********************************/
int main( ) 
{
	//previous Homography
	struct previousH {
		int frameNum;
		Mat homography;
	};
	
	//stopwatch to check the execution time
	ExecutionTime stopWatch(1000);
	//time consumed in processing a frame
	double processingTime;
	
	//vectors to recorde the results
	vector<double> resultTime;
	vector<int> resultFrameLoss;
	
	//instantiate and initialize
	struct previousH pHomography;
	pHomography.frameNum=-1;
	pHomography.homography = Mat::eye(3, 3, CV_64F);
	
	//frame gap threshold
	int thresholdFrameGap = 2;
	
	//current refined homography
	Mat incrementalHomography;
	//current final homography
	Mat H;
	//homography in previous iteration step
	Mat H_p;
	//need refinement?
	bool isRefinement = 1; //iterative way to refine H
	//number of unrecognized frame
	int unrecognizedFrameID = 0;

	/************** read pattern image ********************************/
	//1-dominant orientation is computed, 0-dominant orientation is not computed
	bool flag = 0; 
	//the pattern image
	string imgName_1 = "pattern_toy.jpg";
	//load taregt image
	Mat image_1;
	image_1 = imread(imgName_1, CV_LOAD_IMAGE_GRAYSCALE);
	if(!image_1.data) {
      cout << "fail to load " << imgName_1 << endl;
      return -1;
  }

	Mat mat_show, descriptors_1, descriptors_2, out_img;
	vector<Point2f> obj, scene; 
	//key points
	vector<KeyPoint> keypoints_1, keypoints_2;
	//-- Detect keypoint based on ORB
	ORB orb; /*max number of feature points, default=500*/
	orb(image_1, Mat(), keypoints_1, descriptors_1);		
		//-- Compute LDB descriptors
	LDB ldb(48/*patch size, default=48*/);
	ldb.compute(image_1, keypoints_1, descriptors_1, flag);	
			
	/************** read current frame from video file ***************************/
	//the current frame
	Mat currentFrame;
	//current gray frame
	Mat currentGray;
	//current warpped frame
	Mat image_2;
	//frame id
	int frameID = 0;
	//total frames
	int numFrames;
	//initial homography for case 1)
	Mat initialHomography;
	//if pattern exists in the frame
	int patternExist = 0; //not exist
	//max number of good matched pairs
	int maxGoodMatches = 500;
	//int number of good matched pairs
	int minGoodMatches = 20;
	//the input testing video
	string fileName = "video_toy.avi";
	
	VideoCapture videoSource(fileName);
	if (!videoSource.isOpened()) {
		cout << "Failed to open the video file: " << fileName << endl;
		return -1;
	}
	else {
		numFrames = (int) videoSource.get(CV_CAP_PROP_FRAME_COUNT);
		cout << "Opened video file, there are totally " << numFrames << "frames." 
					<< endl;
	}
	
	//output video
	int askFileTypeBox=0; //-1 is show box of codec  
 	int Color = 1;  
 	Size S = Size( (int)videoSource.get(CV_CAP_PROP_FRAME_WIDTH),   
             (int)videoSource.get(CV_CAP_PROP_FRAME_HEIGHT));  
 	//make output video file  
 	VideoWriter outVideo;  
 		outVideo.open("outVideo_toy.avi", askFileTypeBox, videoSource.get(CV_CAP_PROP_FPS), S, Color); 
 

	//read the frames from the video 
	while (frameID < numFrames) {
		//read a frame
	 	if (!videoSource.read(currentFrame)) {
	 		cout << "Failed to read video frame " << frameID << endl;
	 		return -1;
	 	}
	 	else {
	 		cout << "Successfully loaded frame " << frameID << endl;
	 		frameID++; 		
	 		//convert the current frame into gray scale
	 		if (currentFrame.channels() > 1) {
		      cvtColor(currentFrame, currentGray, CV_RGB2GRAY);
		  }
	 	}

		/************** compare between current frame and pattern image ****************/
		vector<Point2f> obj_corners(4), scene_corners(4);
			obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image_1.cols, 0 );
			obj_corners[2] = cvPoint( image_1.cols, image_1.rows ); obj_corners[3] = cvPoint( 0, image_1.rows );
		//record the starting time		
		stopWatch.tic();

		//case 1): first frame of the game or come back after a while, //case 2): in the middle of the game
		//check previous homography
		if (frameID - pHomography.frameNum > thresholdFrameGap) { //case 1)
			image_2 = currentGray.clone();
		}
		else {					
			//warp current frame
			warpPerspective(currentGray, image_2, pHomography.homography, image_1.size()/*currentGray.size()*/, 
						cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);			
		}

		//calculate homography, if successful  
		if (computeHomography(image_1, keypoints_1, descriptors_1, image_2, flag, minGoodMatches, incrementalHomography)) {
 		
			if (frameID - pHomography.frameNum <= thresholdFrameGap) { //case 2)
				H = pHomography.homography * incrementalHomography;
				//update
				pHomography.homography = H.clone();
				pHomography.frameNum = frameID;
			}
			else {
				H = incrementalHomography.clone();
				//update
				pHomography.homography = H.clone();
				pHomography.frameNum = frameID;			
			}	
			
			patternExist = 1;
			resultFrameLoss.push_back(patternExist);
			
			perspectiveTransform(obj_corners, scene_corners, H);

			// Show detected matches, delay 20ms to show the images
			ostringstream stringStream;
  		stringStream << "Detected a Toy";
  		string copyOfStr = stringStream.str();
			// Show detected matches, delay 20ms to show the images
			putText(currentFrame, copyOfStr, cvPoint(100,100), FONT_HERSHEY_COMPLEX_SMALL, 1.2, cvScalar(0,0,255), 1, CV_AA);
			
			imshow( "Current frame", currentFrame );
			outVideo << currentFrame;
			char key = waitKey(20);
			if (key == 27) // ESC
			 	break;
			}	
			
		else {//recognition failure (no pattern exist)
			patternExist = 0;
			resultFrameLoss.push_back(patternExist);

			imshow( "Current frame", currentFrame );
			
			unrecognizedFrameID++;

			outVideo << currentFrame;

			char key = waitKey(20);
			if (key == 27) // ESC
			 	break;			
		}	
		
		//record the execution time
		processingTime = stopWatch.toc();
		resultTime.push_back(processingTime);
					
		} //end of while

	videoSource.release();
	outVideo.release();

	//print the statistics
	int kk;
	double sumTime = 0.0;
	double numRecognizedFrame = 0;
	cout << "*********************************************" << endl;
	cout << "Frame ID\t" << "processing time (ms)\t" << "pattern exist? (0--No, 1-- Yes)" << endl;
	for(kk = 0; kk < resultTime.size(); kk++) {
		sumTime += resultTime.at(kk);
		numRecognizedFrame += resultFrameLoss.at(kk);
		cout << kk << "\t" << resultTime.at(kk) << "\t" << resultFrameLoss.at(kk) << endl;
	}
	double missingRate = (resultFrameLoss.size() - numRecognizedFrame)/resultFrameLoss.size();
	cout << "The average processing time per frame is " << sumTime/resultTime.size() << "ms." << endl;
	cout << "The ratio of the frames without pattern is " << missingRate << endl;

	/************** end of processing the current frame ****************/
	waitKey(0);
	return 0;
}

