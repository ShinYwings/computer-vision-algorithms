//http://www.gisdeveloper.co.kr/?p=6922
#include <iostream>
#include <cmath>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#endif

using namespace cv;
using namespace cv::xfeatures2d;

using namespace std;
using namespace cv;

Mat drawlines(Mat img1, Mat img2, Mat lines, vector<Point2f>pts1, vector<Point2f> pts2);
Mat FMat(Mat img, vector<Point2f> kp1, vector<Point2f> kp2);
Mat computeEpilines(Mat F, vector<Point2f>kp, int mode);

void normalizePixelCoordinates(const vector<Point2f>& p1, const vector<Point2f>& p2, vector<Point2f>& normVector1, vector<Point2f>& normVector2, Mat& outputMat1, Mat& outputMat2);
vector<Point3f> augmentVector(const vector<Point2f>& inputPoints);
void vectorMean(const vector <Point3f>& inputVector, float& xMean, float& yMean);
void calculateDistance(const vector<Point3f>& inputVector, Mat& outputMat, float xMean, float yMean);
void normalizePoint(const vector<Point3f> vec, const Mat& tr, vector<Point2f>& outputVec);

Mat img1;

int main() {
	img1 = imread("img2.jpeg", IMREAD_GRAYSCALE);
	Mat img2 = imread("img3.jpeg", IMREAD_GRAYSCALE);
	
	cv::resize(img1, img1, cv::Size(800,600));
	cv::resize(img2, img2, cv::Size(800,600));

	if (img1.empty() || img2.empty())
		return -1;

	GaussianBlur(img1, img1, Size(5, 5), 0.0);
	GaussianBlur(img2, img2, Size(5, 5), 0.0);

	vector<cv::KeyPoint> keypoints1;
	Mat descriptors1;

	vector<cv::KeyPoint> keypoints2;
	Mat descriptors2;

	// See all feature2D methods (orb, brisk)
	// https://junsk1016.github.io/opencv/Feature-Matching/
	cv::Ptr<cv::Feature2D> sift = cv::SIFT::create(500);
	
	sift->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
	sift->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
	
	// cv::Ptr<cv::DescriptorMatcher> Matcher_SIFT = cv::BFMatcher::create(cv::NORM_L2);
	vector<vector<DMatch>> matches;
	FlannBasedMatcher matcher;
	int k = 2;
	matcher.knnMatch(descriptors1, descriptors2, matches, k);

	vector<DMatch> goodMatches;
	float nndrRatio = 0.4f;
	for (int i = 0; i < matches.size(); i++) {
		if (matches.at(i).size() == 2 &&
			matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
			goodMatches.push_back(matches[i][0]);
	}

	cout << "goodMatch size: " << goodMatches.size() << endl;

	if (goodMatches.size() < 8)
		return 0;

	Mat imgMatches;
	drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches);
	imshow("goodMatches", imgMatches);
	
	vector<Point2f> kp1;
	vector<Point2f> kp2;
	for (int i = 0; i < goodMatches.size(); i++) {
		kp1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		kp2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	//for(int i)

	//epip
	Mat fundamental_matrix = findFundamentalMat(kp1, kp2, FM_8POINT);//, 3, 0.99); //FM_RANSAC
	cout << "Fundamental_matrix" << endl;
	cout << fundamental_matrix << endl;

	Mat F;
	F = FMat(img1, kp1, kp2);
	cout << "my F matrix" << endl;
	cout << F << endl;

	Mat lines1, lines2;
	//computeCorrespondEpilines(kp2, 2, fundamental_matrix, lines1);
	//computeCorrespondEpilines(kp1, 1, fundamental_matrix, lines2);
	lines1 = computeEpilines(F, kp2, 2);
	lines2 = computeEpilines(F, kp1, 1);

	Mat img3 = drawlines(img1, img2, lines1, kp1, kp2);
	Mat img4 = drawlines(img2, img1, lines2, kp2, kp1);

	imshow("1st", img3);
	imshow("2nd", img4);
	waitKey(0);

	return 0;
}

// Somin's way (weak aligned, but still outputs valid fundamental matrix)
vector<Point2f> NormPoints(vector<Point2f> kp, Mat T) {
		
	T.at<float>(0, 0) = 1.0f / float(img1.cols);
	T.at<float>(0, 2) = -1.0f;
	
	T.at<float>(1, 1) = 1.0f / float(img1.rows);
	T.at<float>(1, 2) = -1.0f;

	T.at<float>(2, 2) = 1.0f;

	int size = kp.size();

	Mat newKp(3, size, CV_32F);
	for (int i = 0; i < kp.size(); i++) {
		newKp.at<float>(0, i) = kp[i].x;
		newKp.at<float>(1, i) = kp[i].y;
		newKp.at<float>(2, i) = 1;
	}

	vector<Point2f> newpt;
	for (int i = 0; i < size; i++) {
		
		Mat pts = T * newKp.col(i);

		
		newpt.push_back(Point2f(pts.at<float>(0), pts.at<float>(1)));
	}
	
	return newpt;
}


Mat FMat(Mat img, vector<Point2f> keypoint1, vector<Point2f> keypoint2) { //kp1, kp2, FM_RANSAC, 3, 0.99

	Mat T1 = Mat::zeros(3, 3, CV_32F);
	Mat T2 = Mat::zeros(3, 3, CV_32F);

	// vector<Point2f> kp1, kp2;
	// normalizePixelCoordinates(keypoint1, keypoint2, kp1, kp2, T1, T2);
	
	vector<Point2f> kp1 = NormPoints(keypoint1, T1);
	vector<Point2f> kp2 = NormPoints(keypoint2, T2);


	int size = kp1.size();
	Mat A(size, 9, CV_32F);
	for (int i = 0; i < kp1.size(); i++) {
		A.at<float>(i, 0) = kp1[i].x * kp2[i].x;
		A.at<float>(i, 1) = kp1[i].y * kp2[i].x;
		A.at<float>(i, 2) = kp2[i].x;

		A.at<float>(i, 3) = kp1[i].x * kp2[i].y;
		A.at<float>(i, 4) = kp1[i].y * kp2[i].y;
		A.at<float>(i, 5) = kp2[i].y;

		A.at<float>(i, 6) = kp1[i].x;
		A.at<float>(i, 7) = kp1[i].y;
		A.at<float>(i, 8) = 1;
	}

	SVD svd(A, SVD::FULL_UV);

	Mat tmp; 
	transpose(svd.vt, tmp);
	
	Mat F;
	F = tmp.col(8);   // get last col vector
	
	Mat F2(3, 3, CV_32F); // row, col
	F2.at<float>(0, 0) = F.at<float>(0, 0);
	F2.at<float>(0, 1) = F.at<float>(1, 0);
	F2.at<float>(0, 2) = F.at<float>(2, 0);
	F2.at<float>(1, 0) = F.at<float>(3, 0);
	F2.at<float>(1, 1) = F.at<float>(4, 0);
	F2.at<float>(1, 2) = F.at<float>(5, 0);
	F2.at<float>(2, 0) = F.at<float>(6, 0);
	F2.at<float>(2, 1) = F.at<float>(7, 0);
	F2.at<float>(2, 2) = F.at<float>(8, 0);


	SVD svdF(F2, SVD::FULL_UV);

	Mat d = Mat::zeros(3, 3, CV_32F);
	
	d.at<float>(0, 0) = svdF.w.at<float>(0);
	d.at<float>(1, 1) = svdF.w.at<float>(1);
	d.at<float>(2, 2) = 0.0;

	Mat fin = svdF.u * d * svdF.vt;
	
	transpose(T2, T2);
	
	fin = T2 * fin * T1;   // normalize 

	float F33 = fin.at<float>(2, 2);
	
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			fin.at<float>(i, j) /= F33;

	return fin;
}


Mat drawlines(Mat img1, Mat img2, Mat lines, vector<Point2f>pts1, vector<Point2f> pts2) {

	Mat newImg(img1.cols, img1.rows, CV_32FC3);
	newImg = img1.clone();
	cv::cvtColor(newImg, newImg, cv::COLOR_GRAY2BGR);
	//cout << lines.at<float>(0, 0) << " " << lines.at<float>(0, 1) << " " << lines.at<float>(0, 2) << endl;
	for (int i = 0; i < lines.rows; i++) {
		int x0 = 0;
		int y0 = int(-lines.at<float>(i, 2) / lines.at<float>(i, 1));
		int x1 = int(img1.cols);
		int y1 = int(-(lines.at<float>(i, 2) + lines.at<float>(i, 0) * img1.cols) / lines.at<float>(i, 1));

		//line(dst, Point(pointers_1[i][0], pointers_1[i][1]), Point(pointers_2[idx][0] + first.cols, pointers_2[idx][1]), (255, 0, 0), 2);
		line(newImg, Point(x0, y0), Point(x1, y1), Scalar(0, 255, 3), 1);
		circle(newImg, Point(pts1[i].x, pts1[i].y), 5, Scalar(0, 0, 255), 1);
		//circle(newImg, pts2[i], 5, (0.255, 3), 1);
	}

	return newImg;
}

Mat computeEpilines(Mat F, vector<Point2f>kp, int mode) {
	/*
	mode == 1
	l2 = F * x1

	mode == 2
	l1 = F.T * x2

	ax + by + c = 0 (a, b, c)
	a^2 + b^2 =1 (normalized)
	*/

	F.convertTo(F, CV_32F);

	int size = kp.size();
	Mat line2(size, 3, CV_32F);
	for (int i = 0; i < size; i++) {
		
		Mat pt(3, 1, CV_32F);
		pt.at<float>(0, 0) = kp[i].x;
		pt.at<float>(1, 0) = kp[i].y;
		pt.at<float>(2, 0) = 1;

		Mat tmp;
		Mat Ft;

		if(mode == 1)
			tmp = F * pt;
		else {
			transpose(F, Ft);
			tmp = Ft * pt;
		}
		
		float a = tmp.at<float>(0, 0);
		float b = tmp.at<float>(1, 0);

		float t = sqrt(pow(a, 2) + pow(b, 2));

		line2.at<float>(i, 0) = a / t;
		line2.at<float>(i, 1) = b / t;
		line2.at<float>(i, 2) = tmp.at<float>(2, 0) / t;
	}

	return line2;
}

// Reference
// https://github.com/erkanoguz/Fundamental-Matrix/blob/master/Fundamental%20Matrix/FundamentalMatrixOperation.cpp
void normalizePixelCoordinates(const vector<Point2f>& p1, const vector<Point2f>& p2, vector<Point2f>& normVector1, vector<Point2f>& normVector2, Mat& outputMat1, Mat& outputMat2)
{
	
	// Point2f to homogenous coordinates
	vector<Point3f> augVec1 = augmentVector(p1);
	vector<Point3f> augVec2 = augmentVector(p2);

	// Find the centroid of image
	float xMean1, yMean1, xMean2, yMean2;
	vectorMean(augVec1, xMean1, yMean1);		// Compute mean of left image point 
	vectorMean(augVec2, xMean2, yMean2);		// Compute mean of right image points

	//cout << "mean of pts1 x: " << xMean1 << " mean of pts1 y: " << yMean1 << endl;
	//cout << "mean of pts2 x: " << xMean2 << " mean of pts2 y: " << yMean2 << endl;

	// transform matrix to normalize the point
	//Mat outputMat1 = Mat::zeros(3, 3, CV_32F);
	//Mat outputMat2 = Mat::zeros(3, 3, CV_32F);

	calculateDistance(augVec1, outputMat1, xMean1, yMean1);
	calculateDistance(augVec2, outputMat2, xMean2, yMean2);

	normalizePoint(augVec1, outputMat1, normVector1);
	normalizePoint(augVec2, outputMat2, normVector2);
}

vector<Point3f> augmentVector(const vector<Point2f>& inputPoints)
{

	// Calculate the number of points
	int numberOfPoints = inputPoints.size();

	// agument the vector with 1
	vector <Point3f> outputVector;
	for (int i = 0; i < numberOfPoints; i++)
	{
		outputVector.push_back(Point3f(inputPoints[i].x, inputPoints[i].y, 1));
	}

	return outputVector;
}

void vectorMean(const vector <Point3f>& inputVector, float& xMean, float& yMean)
{
	// Initializing sum variable
	float sumX = 0;
	float sumY = 0;

	// Sum both x and y points
	for (int i = 0; i < inputVector.size(); i++)
	{
		sumX += (float)inputVector[i].x;
		sumY += (float)inputVector[i].y;
	}

	// Calculate mean
	xMean = sumX / (float)inputVector.size();
	yMean = sumY / (float)inputVector.size();

}

void vector2FMean(const vector <Point2f>& inputVector, float& xMean, float& yMean)
{
	// Initializing sum variable
	float sumX = 0;
	float sumY = 0;

	// Sum both x and y points
	for (int i = 0; i < inputVector.size(); i++)
	{
		sumX += (float)inputVector[i].x;
		sumY += (float)inputVector[i].y;
	}

	// Calculate mean
	xMean = sumX / (float)inputVector.size();
	yMean = sumY / (float)inputVector.size();

}

void calculateDistance(const vector<Point3f>& inputVector, Mat& outputMat, float xMean, float yMean)
{
	float tempVal = 0;
	float tempOut = 0;
	float tempSum = 0;
	float allMean = 0;
	float xSum = 0, ySum = 0;

	Mat temp1 = Mat::zeros(3, 3, CV_32F);
	Mat temp2 = Mat::zeros(3, 3, CV_32F);


	// Aligned zero-centered data
	vector<Point2f> centered_a;

	for (int i = 0; i < inputVector.size(); i++)
	{
		centered_a.push_back(Point2f(inputVector[i].x - xMean, inputVector[i].y - yMean)); //* (xMean - inputVector[i].x);
		
		//tempVal = xTemp + yTemp;
		//tempOut = sqrt(tempVal);
		//tempSum += tempOut;
	}
	
	float centeredMeanX, centeredMeanY;
	vector2FMean(centered_a, centeredMeanX, centeredMeanY);

	for (int i = 0; i < centered_a.size(); i++)
	{
		xSum += pow((centered_a[i].x - centeredMeanX), 2);
		ySum += pow((centered_a[i].y - centeredMeanY) ,2);
	}

	float varX = xSum / centered_a.size();
	float varY = ySum / centered_a.size();

	float std_x = sqrt(varX);
	float std_y = sqrt(varY);

	//allMean = tempSum / (float)inputVector.size();	// Calculate all Mean

	// implement transformation matrix for normalization
	temp1.at<float>(0, 0) = (float)1 / std_x;
	temp1.at<float>(0, 1) = 0;
	temp1.at<float>(0, 2) = 0;
	temp1.at<float>(1, 0) = 0;
	temp1.at<float>(1, 1) = (float)1 / std_y;
	temp1.at<float>(1, 2) = 0;
	temp1.at<float>(2, 0) = 0;
	temp1.at<float>(2, 1) = 0;
	temp1.at<float>(2, 2) = 1;

	temp2.at<float>(0, 0) = 1;
	temp2.at<float>(0, 1) = 0;
	temp2.at<float>(0, 2) = -xMean;
	temp2.at<float>(1, 0) = 0;
	temp2.at<float>(1, 1) = 1;
	temp2.at<float>(1, 2) = -yMean;
	temp2.at<float>(2, 0) = 0;
	temp2.at<float>(2, 1) = 0;
	temp2.at<float>(2, 2) = 1;

	outputMat = temp1 * temp2;
}

void normalizePoint(const vector<Point3f> vec, const Mat& tr, vector<Point2f>& outputVec)
{
	// Convert Point3f to Mat
	Mat inputMat = Mat::zeros(3, vec.size(), CV_32F);

	for (int i = 0; i < vec.size(); i++)
	{
		inputMat.at<float>(0, i) = vec[i].x;
		inputMat.at<float>(1, i) = vec[i].y;
		inputMat.at<float>(2, i) = vec[i].z;
	}

	Mat outputMat = tr * inputMat;

	// Convert Mat to Point2F
	for (int i = 0; i < vec.size(); i++)
	{
		float x = outputMat.at<float>(0, i);
		float y = outputMat.at<float>(1, i);
		outputVec.push_back(Point2f(x, y));
	}

}