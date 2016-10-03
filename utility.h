/**
*******************************************************************************
* @file		PerformanceMetric.h
* @brief	This file provides the interfaces of the classes ExecutionTime and 
functions for computing rms and zncc.
* @author	Yifeng He
* @date		Nov. 11, 2014
*******************************************************************************
**/

#ifndef _PERFORMANCEMETRIC_H_
#define _PERFORMANCEMETRIC_H_

//c++ library
#include <iostream>
#include <time.h>
#include <vector>
#include <algorithm>

//opencv library
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//define the colors
#define CV_RED		Scalar(255,0,0)
#define CV_GREEN	Scalar(0,255,0)
#define CV_BLUE		Scalar(0,0,255)
#define CV_RG			Scalar(255,255,0)
#define CV_RB			Scalar(255,0,255)
#define CV_GB			Scalar(0,255,255)
#define CV_WHITE	Scalar(255,255,255)
#define CV_BLACK	Scalar(0,0,0)
#define CV_GRAY		Scalar(128,128,128)

/**
*******************************************************************************
* @brief		This function draws a polygon by connecting the corners.
* @param 		Mat -- image[int/out], the image to be drawn on
* @param 		vector<Point2f> -- corners [input], the vector of corners of polygon
* @return 	Void
*******************************************************************************
*/ 
inline void drawPolygon(Mat& image, const vector<Point2f>& corners)
{
	Scalar drawColor = Scalar(255, 0, 0);
	int numPoints = corners.size();
	
	//draw lines to connect the corners: point 0 --> 1 --> 2 --> 3 -->... --> 0
	for(int i = 0; i < numPoints; ++i) {
		const Point& r1 = corners[i % 4];
		const Point& r2 = corners[(i + 1) % 4];
		line( image, r1, r2, drawColor, 2 );
	}
}



/**
*******************************************************************************
* @brief		This function draws a quad given the four corners to be mapped by 
Homography.
* @param 		Mat -- image[int/out], the image to be drawn on
* @param 		Mat -- Homo [input], the homography matrix <3x3>
* @param 		Double[][] -- crns [input], the four corners in world coordinate 
system (the left-upper corner is the origin)
* @return 	Void
*******************************************************************************
*/
inline void drawHomography(Mat& image, const Mat& Homo, double const crns[4][2]) 
{
	static Scalar homocolors[] = {
		CV_BLACK,
		CV_GREEN,
		CV_BLUE,
		CV_RED  };
	const Mat_<double>& mH = Homo;
	vector<Point2f> corners(4);
	for (int i = 0; i < 4; i++ ) {
		double ptx = crns[i][0], pty = crns[i][1];
		double w = 1./(mH(2,0)*ptx + mH(2,1)*pty + mH(2,2));
		corners[i] = Point2f((float)((mH(0,0)*ptx + mH(0,1)*pty + mH(0,2))*w),
		            (float)((mH(1,0)*ptx + mH(1,1)*pty + mH(1,2))*w));
	}
	//draw by connecting the four corners in order
	for(int i = 0; i < 4; ++i) {
		const Point& r1 = corners[i%4];
		const Point& r2 = corners[(i+1)%4];
		line( image, r1, r2, homocolors[i], 2 );
	}
	//draw the diagonal lines
	line(image, corners[0], corners[2], CV_GB, 2);
	line(image, corners[1], corners[3], CV_GB, 2);
}



/**
*******************************************************************************
* @brief		This function computs rms error for valid region (non-zero region).
* @param 		Mat -- error[int], an error vector stored in a cv::Mat
* @param 		Mat* -- mask[int], mask->at(i)!=0 means valid region, default is NULL 
pointer which means valid for all.
* @return 	double -- root-mean-squared (rms) error.
*******************************************************************************
*/ 
template <typename MatType>
double computeRMS(const Mat& errorMatrix, const Mat* ptrMask = 0)
{
	//mask matrix in a row
	Mat maskRow;
	//number of valid elements
	int validNumber;
	if (ptrMask) {
		maskRow = ptrMask->reshape(0,1);
		validNumber = countNonZero(maskRow);
	} 
	else {
		validNumber = errorMatrix.total();
	}
	
	/*case 1): if mask==0, maskRow is empty, and validNumber = errorMatrix.total();
	  case 2): if mask !=0, there are 2 sub-cases: case 21): maskRow contains all 
	  zeros, and case 22): maskRow contain at least one element greater than 0*/
	if (!maskRow.empty() && validNumber == 0) //case 21)
		//return NAN (Not A Number)
		return numeric_limits<double>::quiet_NaN();

	//sum of squared errors (sse)
	double sumSquaredError = 0;
	// errors in a row
	Mat errorRow = errorMatrix.reshape(0,1);
	//calculate the sum of sqaured errors
	for(int i = 0; i < (int)errorRow.total(); ++i) {
		MatType errorElement = errorRow.at<MatType>(i);
		/*an element is valid (==1) in the following cases: 1) ptrMask == 0 (default),
		2) ptrMask != 0 and maskRow.at(i) !=0 */
		bool isValid = maskRow.empty() || maskRow.at<MatType>(i) != 0;
		
		sumSquaredError += isValid * (errorElement * errorElement);
	}
	//root-mean-sqared error
	double rms = sqrt (sumSquaredError / validNumber);
	return rms;
}



/**
*******************************************************************************
* @brief		This function computes zero mean normalized cross-correlation (ZNCC) 
between two matrices for valid (non-zero) region specified by mask.
* @param 		Mat -- vector w
* @param 		Mat* -- vector t
* @return 	double -- zncc
*******************************************************************************
*/
template<typename MatType>
double computeZNCC(const Mat& wMatrix, const Mat& tMatrix, const Mat* ptrMask = 0)
{
	//mask in a row
	Mat maskRow;
	//valid number of elements
	int validNumber;
	if (ptrMask) {
		maskRow = ptrMask->reshape(0,1);
		validNumber = countNonZero(maskRow);
	} 
	else {
		validNumber = wMatrix.total();
	}
	
	if (!maskRow.empty() && validNumber == 0) {
		//return NAN (Not A Number)
		return numeric_limits<double>::quiet_NaN();
	}

	Scalar meanW, meanT, stdW, stdT;
	Mat wRow = wMatrix.reshape(0,1);
	Mat tRow = tMatrix.reshape(0,1);
	
	//calculate the means and standard deviations for both matrices without mask
	if (maskRow.empty()) {
		meanStdDev(wRow, meanW, stdW);
		meanStdDev(tRow, meanT, stdT);
	} 
	//calculate the means and standard deviations for both matrices with mask
	else { 
		meanStdDev(wRow, meanW, stdW, maskRow);
		meanStdDev(tRow, meanT, stdT, maskRow);
	}

	//calculate the sum of cross-correlations
	double sumCC = 0;
	for(int i = 0; i < (int)wMatrix.total(); ++i) {
		MatType wElement = wRow.at<MatType>(i);
		MatType tElement = tRow.at<MatType>(i);
		//for each non-zero element in wMatrix, calculate cross-correlation
		sumCC += (wElement != 0) * (wElement - meanW.val[0]) * (tElement - meanT.val[0]);
	}
	//calculate ZNCC
	double zncc = sumCC / (stdW.val[0] * stdT.val[0] * validNumber);
	return zncc;
}



/**
*******************************************************************************
* @brief		This function finds the best center position of the patch 
* @param 		Mat& image -- the frame that contains the target image
* @param 		int cornersTargetImage[4][2] -- the coordinates of the four corners
of the target image in an arbitrary order
* @param 		int blockSize -- the size (in pixels) of the block for which the 
standard deviation will be computed
* @param 		int patchSize -- the size (in pixels) of the patch
* @param 		float result[] -- the result to be resturned to the caller. It contains 
4 elements: result[0] is the row number of the patch center in the whole frame, 
result[1] is the col number of the patch center in the whole frame, result[2] is
the updated patch size in pixels (the patch size will be updated only when the 
input patchSize is larger than the size of the target image), and result[3] is 
the sum of the standard deviations of the blocks in the optimal patch
* @return 	void
*******************************************************************************
*/
inline void findPatchCenter(Mat& image, int cornersTargetImage[4][2], 
															int blockSize, int patchSize, float result[]) 
{
	//check if the inputs are valid
	//the input image should be gray scale
	if (image.channels() > 2) {
		cout << "The input image should be gray scale. Exit..." << endl;
		return;
	}
	
	//crop it into a rectangle area
	int arrayRows[4], arrayCols[4];
	int k;
	for (k = 0; k < 4; k++) {
		arrayRows[k] = cornersTargetImage[k][1]; 
		arrayCols[k] = cornersTargetImage[k][0]; 
	}
	//sort
	sort(arrayRows, arrayRows + 4);
	sort(arrayCols, arrayCols + 4);
			
	//the upper-left corner and bottom-right corner of the cropped area
	int upperLeft[2], bottomRight[2];
	upperLeft[0] = arrayCols[1]; //starting col. number
	upperLeft[1] = arrayRows[1]; //starting row number
	bottomRight[0] = arrayCols[2]; //ending col. number
	bottomRight[1] = arrayRows[2]; //ending row number
	
	//check image boundary
	int protectionMargin = 0;
	if ( (upperLeft[0] > image.cols - protectionMargin) || 
				(upperLeft[1] > image.rows - protectionMargin) ) {
				cout << "Cannot find patch because the target image is outside of the image boundary. Quit..." 
						<< endl;
				return;
	}
	if (bottomRight[0] > image.cols - protectionMargin) {
		bottomRight[0] = image.cols - protectionMargin;
	}
	if (bottomRight[1] > image.rows - protectionMargin) {
		bottomRight[1] = image.rows - protectionMargin;
	}	
		
	//find target image area in the current frame	
	image = image(Range(upperLeft[1] /*strat row number 
								(included)*/, bottomRight[1] /*end row number (excluded)*/), 
								Range(upperLeft[0], bottomRight[0])).clone();
	//check if the patch size is larger than the image size. If larger, change the patch size
	int updatedPatchSize = min(patchSize, min(image.rows, image.cols));
	
	//Mat to store the variance of each block
	int rowsVarMat = image.rows/blockSize;
	int colsVarmat = image.cols/blockSize;
	Mat varMatrix = Mat::zeros(rowsVarMat, colsVarmat, CV_32F); 
	
	//calculate the variance for each block
	int i, j;
	Mat tempBlock;
	//the mean of the block
	Scalar meanBlock;
	//the standard deviation of the block
	Scalar stdBlock; 
	for (i = 0; i < rowsVarMat; i++) {
		for (j = 0; j < colsVarmat; j++) {
			//calculate the variance and put it in varMatrix(i,j)
			tempBlock = image(Range(blockSize*i, blockSize*(i+1)),
				Range(blockSize*j, blockSize*(j+1)) ); 
			meanStdDev(tempBlock, meanBlock, stdBlock);
			varMatrix.at<float>(i,j) = stdBlock[0];
		}
	}
	
	//number of blocks in the patch
	int numBlocks = updatedPatchSize/blockSize;
	//Mat to record the standard deviations for the blocks
	int rowsMatPatch = varMatrix.rows - numBlocks + 1;
	int colsMatPatch = varMatrix.cols - numBlocks + 1;
	Mat matPatch = Mat::zeros(rowsMatPatch, colsMatPatch, CV_32F);;
	
	//the sum of the standard deviations for the blocks in the patch
	float stdSum;
	//array to store the max sum of variance
	float maxStd[3] = {0, 0, 0}; //row, col, sum_std
	//find the max sum of the standard deviations
	for (i = 0; i < rowsMatPatch; i++) {
		for (j=0; j < colsMatPatch; j++) {
			matPatch = varMatrix(Range(i, i+numBlocks), Range(j, j+numBlocks));
			stdSum = sum(matPatch)[0];
			//record the max standard deviation
			if (stdSum > maxStd[2]) {
				maxStd[0] = i; //row No.
				maxStd[1] = j; //col No.
				maxStd[2] = stdSum;
			}		
		}
	}
	
	//find the upper left conner of the optimal patch 
	int rowPatch = blockSize*maxStd[0]; //row
	int colPatch = blockSize*maxStd[1]; //col

#ifdef DEBUG		
//	//draw the optimal patch (for debug)
//	vector<Point2f> cornersPatch;
//	Point2f upperLeftPatch(colPatch, rowPatch);
//	Point2f upperRightPatch(colPatch, rowPatch+updatedPatchSize);
//	Point2f bottomLeftPatch(colPatch+updatedPatchSize, rowPatch);
//	Point2f bottomRightPatch(colPatch+updatedPatchSize, rowPatch+updatedPatchSize);	
//	cornersPatch.push_back(upperLeftPatch);
//	cornersPatch.push_back(upperRightPatch);
//	cornersPatch.push_back(bottomLeftPatch);
//	cornersPatch.push_back(bottomRightPatch);
//	drawPolygon(image, cornersPatch);
//	imshow("function: image with Patch", image);
#endif
	
	//the center of the patch in the target image
	int rowCenterPatch = rowPatch + updatedPatchSize/2;
	int colCenterPatch = colPatch + updatedPatchSize/2;
	
	//the row number of the patch center in the whole frame
	result[0] = rowCenterPatch + upperLeft[1];
	//the col number of the patch center in the whole frame
	result[1] = colCenterPatch + upperLeft[0];
	//the updated patch size
	result[2]	= updatedPatchSize;
	//the sum of the standard deviations of the blocks in the optimal patch 
	result[3] = maxStd[2];
}



/**
*******************************************************************************
* @brief		This function uses the image center as the patch center 
* @param 		Mat& image -- the frame that contains the target image
* @param 		int cornersTargetImage[4][2] -- the coordinates of the four corners
of the target image in an arbitrary order
* @param 		int blockSize -- the size (in pixels) of the block for which the 
standard deviation will be computed
* @param 		int patchSize -- the size (in pixels) of the patch
* @param 		float result[] -- the result to be resturned to the caller. It contains 
4 elements: result[0] is the row number of the patch center in the whole frame, 
result[1] is the col number of the patch center in the whole frame, result[2] is
the updated patch size in pixels (the patch size will be updated only when the 
input patchSize is larger than the size of the target image), and result[3] is 
the sum of the standard deviations of the blocks in the optimal patch
* @return 	void
*******************************************************************************
*/
inline void findPatchCenterFast(Mat& image, int cornersTargetImage[4][2], 
															int blockSize, int patchSize, float result[])  
{	
	//check if the inputs are valid
	//the input image should be gray scale
	if (image.channels() > 2) {
		cout << "The input image should be gray scale. Exit..." << endl;
		return;
	}
	
	//crop it into a rectangle area
	int arrayRows[4], arrayCols[4];
	int k;
	for (k = 0; k < 4; k++) {
		arrayRows[k] = cornersTargetImage[k][1]; 
		arrayCols[k] = cornersTargetImage[k][0]; 
	}
	//sort
	sort(arrayRows, arrayRows + 4);
	sort(arrayCols, arrayCols + 4);
			
	//the upper-left corner and bottom-right corner of the cropped area
	int upperLeft[2], bottomRight[2];
	upperLeft[0] = arrayCols[1]; //starting col. number
	upperLeft[1] = arrayRows[1]; //starting row number
	bottomRight[0] = arrayCols[2]; //ending col. number
	bottomRight[1] = arrayRows[2]; //ending row number
	
	//check image boundary
	int protectionMargin = 0;
	if ( (upperLeft[0] > image.cols - protectionMargin) || 
				(upperLeft[1] > image.rows - protectionMargin) ) {
				cout << "Cannot find patch because the target image is outside of the image boundary. Quit..." 
						<< endl;
				return;
	}
	if (bottomRight[0] > image.cols - protectionMargin) {
		bottomRight[0] = image.cols - protectionMargin;
	}
	if (bottomRight[1] > image.rows - protectionMargin) {
		bottomRight[1] = image.rows - protectionMargin;
	}
		
	//find target image area in the current frame
	image = image(Range(upperLeft[1] /*strat row number 
								(included)*/, bottomRight[1] /*end row number (excluded)*/), 
								Range(upperLeft[0], bottomRight[0])).clone();
	//find the image center
	int imgCenterRow = image.rows/2;
	int imgCenterCol = image.cols/2;
	
	//check if the patch size is larger than the image size. If larger, change the patch size
	int updatedPatchSize = min(patchSize, min(image.rows, image.cols));	
	
	//find the upper left conner of the optimal patch 
	int rowPatch = imgCenterRow - updatedPatchSize/2; //row
	int colPatch = imgCenterCol - updatedPatchSize/2; //col

#ifdef DEBUG		
//	//draw the optimal patch (for debug)
//	vector<Point2f> cornersPatch;
//	Point2f upperLeftPatch(colPatch, rowPatch);
//	Point2f upperRightPatch(colPatch, rowPatch+updatedPatchSize);
//	Point2f bottomLeftPatch(colPatch+updatedPatchSize, rowPatch);
//	Point2f bottomRightPatch(colPatch+updatedPatchSize, rowPatch+updatedPatchSize);	
//	cornersPatch.push_back(upperLeftPatch);
//	cornersPatch.push_back(upperRightPatch);
//	cornersPatch.push_back(bottomLeftPatch);
//	cornersPatch.push_back(bottomRightPatch);
//	drawPolygon(image, cornersPatch);
//	imshow("function: image with Patch", image);
#endif
	
	//the center of the patch in the target image
	int rowCenterPatch = rowPatch + updatedPatchSize/2;
	int colCenterPatch = colPatch + updatedPatchSize/2;
	
	//the row number of the patch center in the whole frame
	result[0] = rowCenterPatch + upperLeft[1];
	//the col number of the patch center in the whole frame
	result[1] = colCenterPatch + upperLeft[0];
	//the updated patch size
	result[2]	= updatedPatchSize;
	//the sum of the standard deviations of the blocks in the optimal patch 
	result[3] = 0;
}


/**
*******************************************************************************
* @class		ExecutionTime
* @brief 		This class defines the interfaces for the class ExecutionTime.
*******************************************************************************
*/
class ExecutionTime
{
	private:
		//scale: 1 means second, 1000 means milli-second, 1/60.0 means minutes
		int scale_;
		//starting tick
		double startTick_;



	public:
		/**
		***************************************************************************
		* @brief		This function is the constructor for class ExecutionTime. 
		* @param		Int -- the scale (default 1)
		* @return		None
		***************************************************************************
		*/
		ExecutionTime(int scale = 1) {
			this->scale_ = scale;
			startTick_ = 0;
		}


	
		/**
		***************************************************************************
		* @brief		This function starts the stopwatch. 
		* @param		None
		* @return		Double -- the current tick
		***************************************************************************
		*/
		inline double tic() {
			return startTick_ = (double)getTickCount();
		}


	
		/**
		***************************************************************************
		* @brief		This function stops the stopwatch. 
		* @param		None
		* @return		Double -- the time interval
		***************************************************************************
		*/
		inline double toc() {
			return ((double)getTickCount() - startTick_)/getTickFrequency() * scale_;
		}


	
		/**
		***************************************************************************
		* @brief		This function stops the stopwatch and re-starts it immediately. 
		* @param		None
		* @return		Double -- the time interval
		***************************************************************************
		*/
		inline double toctic() {
			double interval = ((double)getTickCount() - startTick_)/getTickFrequency() 
												* scale_;
			//re-start the stopwatch
			tic();
			return interval;
		}
	
}; //end of class ExecutionTime

#endif //_PERFORMANCEMETRIC_H_



