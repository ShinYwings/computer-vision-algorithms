#pragma once
#pragma once
#pragma once
#pragma once
#pragma once
#pragma once

//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <string>
//#include <algorithm>
#include <math.h>
#include <time.h>
#include <iomanip>


#define _USE_MATH_DEFINES //<cmath>에서 M_PI 사용하려고...
#include <cmath> 
using namespace std;
#include "DataInterface.h"

#include "voxel_DB.h"
#include "kmeans.h"

#include <opencv2\opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2\cudafeatures2d.hpp>

#define NOMINMAX  //windows.h 헤더 파일에서 min과 max를 전처리기 매크로로 정의를 해서 발생하는 문제를 해결하기 위함.

//#if _DEBUG
//	#pragma comment(lib, "opencv_world343d.lib")
//#endif
//	#pragma comment(lib, "opencv_world343.lib")
#if _DEBUG
#pragma comment(lib, "opencv_world345d.lib")
#endif
#pragma comment(lib, "opencv_world345.lib")
using namespace cv;


vector<string> split(string str, char delimiter) {
	vector<string> internal;
	stringstream ss(str);
	string temp;

	while (getline(ss, temp, delimiter)) {
		internal.push_back(temp);
	}

	return internal;
}


namespace db_points {

	template<typename T>
	void writeVector(ostream &out, const vector<T> &vec)
	{
		//out << vec.size();
		write_typed_data(out, vec.size());

		for (typename vector<T>::const_iterator i = vec.begin(); i != vec.end(); ++i)
		{
			out << *i;
		}
	}


	template<typename T>
	void readVector(istream &in, vector<T> &vec)
	{
		size_t size;
		//in >> size;
		read_typed_data(in, size);
		vec.reserve(size);

		for (int i = 0; i < size; ++i)
		{
			T tmp;
			in >> tmp;
			vec.push_back(tmp);
		}
	}

	ostream &operator << (ostream &out, const DB_Point &i)
	{
		//out << i.x_2d << i.y_2d;  // FIXME Read/write strings properly.
		//out << i.x << i.y << i.z;
		//out << i.clusterID;
		//out << i.pt_2d_id;
		//out << i.pt_3d_id;

		write_typed_data(out, i.x_2d); write_typed_data(out, i.y_2d);
		write_typed_data(out, i.x); write_typed_data(out, i.y); write_typed_data(out, i.z);
		write_typed_data(out, i.clusterID);
		write_typed_data(out, i.pt_2d_id);
		write_typed_data(out, i.pt_3d_id);

		return out;
	}

	istream &operator >> (istream &in, DB_Point &i)
	{
		//// Keep in same order as operator<<(ostream &, const DB_Point &)!
		//in >> i.x_2d >> i.y_2d;  // FIXME Read/write strings properly.
		//in >> i.x >> i.y >> i.z;
		//in >> i.clusterID;
		//in >> i.pt_2d_id;
		//in >> i.pt_3d_id;

		read_typed_data(in, i.x_2d); read_typed_data(in, i.y_2d);
		read_typed_data(in, i.x); read_typed_data(in, i.y); read_typed_data(in, i.z);
		read_typed_data(in, i.clusterID);
		read_typed_data(in, i.pt_2d_id);
		read_typed_data(in, i.pt_3d_id);

		return in;
	}
}







//File loader supports .nvm format and bundler format
bool LoadModelFile(string& name, string& dataname, vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
	vector<string>& names, vector<int>& ptc, int cubeSize);
void SaveNVM(const char* filename, vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
	vector<string>& names, vector<int>& ptc);
void SaveBundlerModel(const char* filename, vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx);

//////////////////////////////////////////////////////////////////
void AddNoise(vector<CameraT>& camera_data, vector<Point3D>& point_data, float percent);
void AddStableNoise(vector<CameraT>& camera_data, vector<Point3D>& point_data,
	const vector<int>& ptidx, const vector<int>& camidx, float percent);
bool RemoveInvisiblePoints(vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<int>& ptidx, vector<int>& camidx,
	vector<Point2D>& measurements, vector<string>& names, vector<int>& ptc);

void GetQuaternionRotationByPnP(cv::Mat R_matrix, cv::Mat t_matrix, float q2[4]);
bool EstimatePoseByPnP(vector<Point2d> features_2d, vector<Point3d> features_3d, double focal_length,
	cv::Mat &A_matrix, cv::Mat &R_matrix, cv::Mat &t_matrix);


bool EstimatePoseByPnP(vector<Point2d> list_points2d, vector<Point3d> list_points3d, double focal_length,
	cv::Mat &A_matrix, cv::Mat &R_matrix, cv::Mat &t_matrix)
{
	double f = focal_length;
	const double params[] = { f,   // fx
							  f,  // fy
							  0,      // cx	
							  0 };    // cy

	A_matrix.at<double>(0, 0) = params[0];       //      [ fx   0  cx ]
	A_matrix.at<double>(1, 1) = params[1];       //      [  0  fy  cy ]
	A_matrix.at<double>(0, 2) = params[2];       //      [  0   0   1 ]
	A_matrix.at<double>(1, 2) = params[3];
	A_matrix.at<double>(2, 2) = 1;

	//micro 웹캠... 왜곡계수
	//double k1 = 0.022774;
	//double k2 = -0.041311;
	//double p1 = -0.0055;
	//double p2 = -0.0009367;
	double k1 = 0;
	double k2 = -0;
	double p1 = -0;
	double p2 = -0;
	double d[] = { k1, k2, p1, p2 };
	cv::Mat distCoeffs(4, 1, CV_64FC1, d);

	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
	bool useExtrinsicGuess = true;
	//bool correspondence = cv::solvePnP(list_points3d, list_points2d, A_matrix, distCoeffs, rvec, tvec,
	//	useExtrinsicGuess, SOLVEPNP_ITERATIVE);
	bool correspondence = cv::solvePnPRansac(list_points3d, list_points2d, A_matrix, distCoeffs, rvec, tvec, useExtrinsicGuess);

	if (correspondence)
	{
		//R|t TEST
		Rodrigues(rvec, R_matrix);
		t_matrix = tvec;

		return correspondence;
	}
	else
		return correspondence;
}


void GetQuaternionRotationByPnP(cv::Mat R_matrix, cv::Mat t_matrix, float q[4])
{
	float r00 = R_matrix.at<double>(0, 0);
	float r01 = R_matrix.at<double>(0, 1);
	float r02 = R_matrix.at<double>(0, 2);
	float r10 = R_matrix.at<double>(1, 0);
	float r11 = R_matrix.at<double>(1, 1);
	float r12 = R_matrix.at<double>(1, 2);
	float r20 = R_matrix.at<double>(2, 0);
	float r21 = R_matrix.at<double>(2, 1);
	float r22 = R_matrix.at<double>(2, 2);
	float t1 = t_matrix.at<double>(0);
	float t2 = t_matrix.at<double>(1);
	float t3 = t_matrix.at<double>(2);

	q[0] = 1 + r00 + r11 + r22;
	if (q[0] > 0.000000001)
	{
		q[0] = sqrt(q[0]) / 2.0;
		q[1] = (r21 - r12) / (4.0 *q[0]);
		q[2] = (r02 - r20) / (4.0 *q[0]);
		q[3] = (r10 - r01) / (4.0 *q[0]);
	}
	else
	{
		double s;
		if (r00 > r11 && r00 > r22)
		{
			s = 2.0 * sqrt(1.0 + r00 - r11 - r22);
			q[1] = 0.25 * s;
			q[2] = (r01 + r10) / s;
			q[3] = (r02 + r20) / s;
			q[0] = (r12 - r21) / s;
		}
		else if (r11 > r22)
		{
			s = 2.0 * sqrt(1.0 + r11 - r00 - r22);
			q[1] = (r01 + r10) / s;
			q[2] = 0.25 * s;
			q[3] = (r12 + r21) / s;
			q[0] = (r02 - r20) / s;
		}
		else
		{
			s = 2.0 * sqrt(1.0 + r22 - r00 - r11);
			q[1] = (r02 + r20) / s;
			q[2] = (r12 + r21) / s;
			q[3] = 0.25f * s;
			q[0] = (r01 - r10) / s;
		}
	}
}

float CalcMHWScore(std::vector<float> hWScores) {
	assert(!hWScores.empty());
	const auto middleItr = hWScores.begin() + hWScores.size() / 2;
	std::nth_element(hWScores.begin(), middleItr, hWScores.end());
	if (hWScores.size() % 2 == 0) {
		const auto leftMiddleItr = std::max_element(hWScores.begin(), middleItr);
		return (*leftMiddleItr + *middleItr) / 2;
	}
	else {
		return *middleItr;
	}
};

/////////////////////////////////////////////////////////////////////////////
bool LoadNVM(ifstream& in, string& dataname, vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
	vector<string>& names, vector<int>& ptc, int cubeSize)
{


	int rotation_parameter_num = 4;
	bool format_r9t = false;
	string token;
	if (in.peek() == 'N')
	{
		in >> token; //file header
		if (strstr(token.c_str(), "R9T"))
		{
			rotation_parameter_num = 9;    //rotation as 3x3 matrix
			format_r9t = true;
		}
	}

	float f_fx = 0, f_fy = 0;
	float f_cx = 0, f_cy = 0;
	if (dataname != "ArtCenter/")
	{
		string FixedK;
		in >> FixedK;

		in >> f_fx >> f_cx >> f_fy >> f_cy;
		float distort = 0;
		in >> distort;

	}

	int ncam = 0, npoint = 0, nproj = 0;
	// read # of cameras
	in >> ncam;  if (ncam <= 1) return false;

	printf("ncam:%d\n", ncam);

	//read the camera parameters
	camera_data.resize(ncam); // allocate the camera data
	names.resize(ncam);
	float radial_distortion = 0;
	float sum_f = 0;
	float sum_distor = 0;
	for (int i = 0; i < ncam; ++i)
	{
		double f, q[9], c[3], d[2];
		in >> token >> f;
		sum_f += f;

		for (int j = 0; j < rotation_parameter_num; ++j) in >> q[j];
		in >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];

		//q는 쿼터니언, c는 translation
		camera_data[i].SetFocalLength(f);
		if (format_r9t)
		{
			camera_data[i].SetMatrixRotation(q);
			camera_data[i].SetTranslation(c);
		}
		else
		{
			//older format for compability -> 쿼터니언과 카메라 중심을 R과 T로 변환 시켜줌
			camera_data[i].SetQuaternionRotation(q);        //quaternion from the file
			camera_data[i].SetCameraCenterAfterRotation(c); //camera center from the file
		}
		camera_data[i].SetNormalizedMeasurementDistortion(d[0]);
		names[i] = token;
		radial_distortion = d[0];
		sum_distor += radial_distortion;
	}

	printf("avgf:%f, avgdistor:%f\n", sum_f / ncam, sum_distor / ncam);

	//////////////////////////////////////
	in >> npoint;   if (npoint <= 0) return false;

	//read image projections and 3D points.
	vector<DB_Point_3D> vec_db_pt_3d;
	vector<vector<Point2f>> features_2d;
	vector<vector<Point3f>> features_3d;
	vector<Point3d> test_features_3d;
	//featureImages.resize(ncam);
	features_2d.resize(ncam);
	features_3d.resize(ncam);
	point_data.resize(npoint);
	for (int i = 0; i < npoint; ++i)
	{
		DB_Point_3D db_pt_3d;

		float pt[3]; int cc[3], npj;
		in >> pt[0] >> pt[1] >> pt[2]
			>> cc[0] >> cc[1] >> cc[2] >> npj;

		db_pt_3d.setXYZ(pt[0], pt[1], pt[2]);
		db_pt_3d.pt_3d_id = i;

		for (int j = 0; j < npj; ++j)
		{
			int cidx, fidx; float imx, imy;
			in >> cidx >> fidx >> imx >> imy;

			camidx.push_back(cidx);    //camera index
			ptidx.push_back(fidx);        //point index

			//add a measurment to the vector
			measurements.push_back(Point2D(imx, imy));
			nproj++;

			//float pt_2d[2] = { 0 };
			//camera_data[cidx].GetProjectionPoint(pt, pt_2d);

			features_2d[cidx].push_back(Point2f(imx, imy));
			features_3d[cidx].push_back(Point3f(pt[0], pt[1], pt[2]));

			db_pt_3d.vec_2d_pt.push_back(Point_2D(imx, imy));
			db_pt_3d.pt_2d_ids.push_back(fidx);
			db_pt_3d.img_IDs.push_back(cidx);

		}

		point_data[i].SetPoint(pt);
		ptc.insert(ptc.end(), cc, cc + 3);

		vec_db_pt_3d.push_back(db_pt_3d);
	}

	//bool asdf = RemoveInvisiblePoints(camera_data, point_data, ptidx, camidx, measurements, names, ptc);

	printf("finish!\n");


	/*
	DBSCAN Clustering
	*/
#define MINIMUM_POINTS 80    // minimum number of cluster
	//#define EPSILON (0.75*0.75)  // distance for clustering, metre^2
#define EPSILON (0.08)  // distance for clustering, metre^2

	vector<DB_Point> points;
	int nLabel = 0;

	npoint = point_data.size();

	//#define NUM_COLOR 300
	srand(0);
	float color[2000][3] = { 0 };
	for (int i = 0; i < 2000; i++)
	{
		color[i][0] = (rand() % 255);
		color[i][1] = (rand() % 255);
		color[i][2] = (rand() % 255);
	}


	float reSize_H = 168, reSize_W = 304;

	VOXEL_DB ds(MINIMUM_POINTS, EPSILON, vec_db_pt_3d, cubeSize);
	//printf("npoint:%d, points.size():%d\n", npoint, points.size());
	//ds.printResults(points, points.size(), true, color);

	ds.voxelFitting3(ds.m_points, ds.getTotalPointSize(), color, nLabel);  //만들어야함.

	int num_points = ds.m_points.size();
	printf("\n num_points:%d\n", num_points);

	vector<DB_Point_3D> tmp_points;
	for (int i = 0; i < num_points; ++i)
	{
		if (ds.m_points[i].clusterID != -1)
		{
			tmp_points.push_back(ds.m_points[i]);
		}
	}
	printf("tmp_points size:%d\n", tmp_points.size());
	printf("nLabel:%d\n", nLabel);

	/*
	label별로 묶기
	*/
	vector<Point3f> vec_centroid_points;
	vector<vector<DB_Point_3D>> vec_label_points;
	vec_label_points.resize(nLabel);

	printf("-------------------------------------------\n");
	for (int i = 0; i < num_points; ++i)
	{
		int label = tmp_points[i].clusterID - 1;
		vec_label_points[label].push_back(tmp_points[i]);
	}

	/*
	plane fitting으로 inlier만 뽑아내서 만들기.
	*/
	vector<DB_Point_3D> vec_inlierData;
	float sumFit_distErr = 0;
	vector<float> vec_meanErr;
	for (int i = 0; i < nLabel; ++i)
	{
		vector<DB_Point_3D> tmp_inlierData;
		float meanErr = ds.PlaneFitting(vec_label_points[i], vec_label_points[i].size(), tmp_inlierData, color);
		vec_meanErr.push_back(meanErr);
		sumFit_distErr += meanErr;
		for (int n = 0; n < tmp_inlierData.size(); ++n)
		{
			vec_inlierData.push_back(tmp_inlierData[n]);
		}

		// 벡터 복사
		vec_label_points[i].clear();
		vec_label_points[i].resize(tmp_inlierData.size());
		copy(tmp_inlierData.begin(), tmp_inlierData.end(), vec_label_points[i].begin());

	}


	float medianErr = CalcMHWScore(vec_meanErr);

	float meanErr = sumFit_distErr / nLabel;
	float variance = 0;
	float sum = 0;
	for (int i = 0; i < vec_meanErr.size(); i++) {
		sum += (meanErr - vec_meanErr[i])*(meanErr - vec_meanErr[i]);
	}
	variance = sum / vec_meanErr.size();
	printf("avgFit_distErr:%f, variance:%f, std:%f, medianErr:%f\n", meanErr, variance, sqrt(variance), medianErr);


	printf("vec_inlierData:%d\n", vec_inlierData.size());
	ds.printResults(vec_inlierData, vec_inlierData.size(), true, color);




	//int max_pt_3d_id = 0;
	//for (int k = 0; k < vec_label_points.size(); ++k)
	//{
	//	for (int l = 0; l < vec_label_points[k].size(); ++l)
	//	{
	//		int pt_3d_id = vec_label_points[k][l].pt_3d_id;
	//		if (pt_3d_id > max_pt_3d_id)
	//			max_pt_3d_id = pt_3d_id;
	//	}
	//}
	//printf("max_pt_3d_id:%d\n", max_pt_3d_id);
	//return false;





	printf("--------- centroid points extracting... -----------\n");
	int sumEachVoxelPtCnt = 0;
	for (int i = 0; i < nLabel; ++i)
	{
		float sumX = 0, sumY = 0, sumZ = 0;
		float avrX = 0, avrY = 0, avrZ = 0;
		for (int j = 0; j < vec_label_points[i].size(); ++j)
		{
			float x = vec_label_points[i][j].x;
			float y = vec_label_points[i][j].y;
			float z = vec_label_points[i][j].z;

			sumX += x; sumY += y; sumZ += z;
		}
		avrX = sumX / vec_label_points[i].size();
		avrY = sumY / vec_label_points[i].size();
		avrZ = sumZ / vec_label_points[i].size();

		Point3f centroidPt;
		int centId = -1;
		float minDist = 99999;
		for (int j = 0; j < vec_label_points[i].size(); ++j)
		{
			float x = vec_label_points[i][j].x;
			float y = vec_label_points[i][j].y;
			float z = vec_label_points[i][j].z;

			float dist = sqrt(pow(avrX - x, 2) + pow(avrY - y, 2) + pow(avrZ - z, 2));
			if (dist < minDist)
			{
				minDist = dist;
				centId = j;
			}
		}

		centroidPt.x = vec_label_points[i][centId].x;
		centroidPt.y = vec_label_points[i][centId].y;
		centroidPt.z = vec_label_points[i][centId].z;
		//centroidPt.x = avrX;
		//centroidPt.y = avrY;
		//centroidPt.z = avrZ;

		vec_centroid_points.push_back(centroidPt);

		sumEachVoxelPtCnt += vec_label_points[i].size();
		printf("centId:%d, minDist:%f, vec_label_points[i]:%d\n", centId, minDist, vec_label_points[i].size());
	}
	int avgEachVoxelPtCnt = sumEachVoxelPtCnt / nLabel;
	printf("sumEachVoxelPtCnt:%d, avgEachVoxelPtCnt:%d\n", sumEachVoxelPtCnt, avgEachVoxelPtCnt);
	printf("vec_centroid_points:%d\n", vec_centroid_points.size());

	/*
	Write 3d centroid points, _voxelFeatureMap_Aachen
	*/
	string save_path = "F:/_voxelFeatureMap_ChungAngUni_Scenes/pair_Voxel_set/" + dataname + "_3d_nearest_centroid.txt";
	ofstream output(save_path);
	output << vec_centroid_points.size() << "\n";
	for (int i = 0; i < vec_centroid_points.size(); ++i)
	{
		float x = vec_centroid_points[i].x;
		float y = vec_centroid_points[i].y;
		float z = vec_centroid_points[i].z;

		output << x << " " << y << " " << z << "\n";
	}
	output.close();



	/*
	DB Point DATA 생성~~~
	*/


	//이미지별로 불러오기
	vector<DB_Image2> vec_db_image;
	for (int i = 0; i < ncam; ++i)
	{
		string img_path = "F:/_ChungAngUniv/" + dataname + "ALL/" + names[i];
		cout << img_path << endl;

		//int nExt = img_path.rfind("jpg");
		//int nName = img_path.rfind("/") + 1;
		//string strModExt("png");
		//string strReName;
		//strReName = img_path.substr(0, nName);
		//strReName += img_path.substr(nName, nExt - nName);
		//strReName += strModExt;

		Mat img = imread(img_path);

		int c_w = img.cols / 2;
		int c_h = img.rows / 2;

		float fx = camera_data[i].f;
		float fy = camera_data[i].f;


		if (dataname != "ArtCenter/")
		{
			fx = f_fx;
			fy = f_fy;

			c_w = f_cx;
			c_h = f_cy;
		}

		//Mat down_img = img.clone();
		//resize(down_img, down_img, Size(reSize_W, reSize_H));  //455, 256

		/*
		clustering 3D Points와 기존 이미지별 들어오는 3D Point와 같은 곳에서,
		각 이미지별 들어오는 각 3D Point에 clustering 3D Points의 라벨을 부여.
		그리고 라벨별로 2d Point를 다시 묶음.
		-> 그 라벨 클러스터에 포인트 개수가 적으면, 특징이 그만큼 적은 것이므로, 이후에 patch로서 볼때 제외 시키기 위함.
		*/
		vector<vector<DB_Point>> eachLabelPt;
		int ss = vec_label_points.size();
		eachLabelPt.resize(ss);
		for (int k = 0; k < vec_label_points.size(); ++k)
		{
			for (int l = 0; l < vec_label_points[k].size(); ++l)
			{
				int imgs_size = vec_label_points[k][l].img_IDs.size();
				for (int m = 0; m < imgs_size; ++m)
				{
					if (vec_label_points[k][l].img_IDs[m] == i)
					{
						float feature_x = vec_label_points[k][l].vec_2d_pt[m].x + c_w + 0.5;
						float feature_y = vec_label_points[k][l].vec_2d_pt[m].y + c_h + 0.5;

						feature_x = feature_x / ((float)img.cols / reSize_W);
						feature_y = feature_y / ((float)img.rows / reSize_H);

						if (feature_x<0 || feature_x>reSize_W)
						{
							printf("feature_x:%d\n", feature_x);
							continue;
						}

						if (feature_y<0 || feature_y>reSize_H)
						{
							printf("feature_y:%d\n", feature_y);
							continue;
						}

						//circle(down_img, Point(feature_x, feature_y), 1, Scalar(255,0,0), -1);

						DB_Point tmp_db;
						int color_lb = vec_label_points[k][l].clusterID - 1;
						tmp_db.clusterID = color_lb + 1;  //배경을 0 라벨로 주기위해... 
						tmp_db.x = vec_label_points[k][l].x;
						tmp_db.y = vec_label_points[k][l].y;
						tmp_db.z = vec_label_points[k][l].z;
						tmp_db.x_2d = feature_x;
						tmp_db.y_2d = feature_y;
						tmp_db.pt_3d_id = vec_label_points[k][l].pt_3d_id;
						tmp_db.pt_2d_id = vec_label_points[k][l].pt_2d_ids[m];

						eachLabelPt[color_lb].push_back(tmp_db);
					}

				}

			}

		}
		//imshow("image GT pts", down_img);
		//waitKey();

		DB_Image2 tmp_db_img;

		float q[4]; float t[3];
		camera_data[i].GetQuaternionRotation(q);
		camera_data[i].GetCameraCenter(t);
		//printf("q:%f %f %f %f, t:%f %f %f\n", q[0], q[1], q[2], q[3], t[0], t[1], t[2]);

		tmp_db_img.quat[0] = q[0]; tmp_db_img.quat[1] = q[1]; tmp_db_img.quat[2] = q[2]; tmp_db_img.quat[3] = q[3];
		tmp_db_img.camCent[0] = t[0]; tmp_db_img.camCent[1] = t[1]; tmp_db_img.camCent[2] = t[2];
		tmp_db_img.img_ID = i;

		string reName = names[i];
		reName.replace(reName.find("jpg"), 3, "png");
		tmp_db_img.img_path = reName;
		//tmp_db_img.img_path = names[i];


		double scaleX = (float)img.cols / reSize_W;
		double scaleY = (float)img.rows / reSize_H;

		tmp_db_img.focal_lenX = fx / scaleX;
		tmp_db_img.focal_lenY = fy / scaleY;
		tmp_db_img.Cx = c_w / scaleX;
		tmp_db_img.Cy = c_h / scaleY;

		tmp_db_img.voxel_db_pt = eachLabelPt;
		//for (int j = 0; j < eachLabelPt.size(); ++j)
		//{
		//	if (eachLabelPt[j].size() < 5)       //각 라벨에 들어오는 포인트의 개수가 5개 미만이면, 제거           
		//		continue;
		//	tmp_db_img.voxel_db_pt.push_back(eachLabelPt[j]);
		//}
		vec_db_image.push_back(tmp_db_img);
	}

	//std::ofstream oStream_first("db_data_first.bin", ios::out | ios::binary); //ios::app => 이어쓰기
	//std::ofstream oStream_second("db_data_second.bin", ios::out | ios::binary); //ios::app => 이어쓰기
	std::ofstream oStream_first("F:/_voxelFeatureMap_ChungAngUni_Scenes/pair_Voxel_set/" + dataname + "db_data_first.txt", ios::out | ios::binary); //ios::app => 이어쓰기
	std::ofstream oStream_second("F:/_voxelFeatureMap_ChungAngUni_Scenes/pair_Voxel_set/" + dataname + "db_data_second.txt", ios::out | ios::binary); //ios::app => 이어쓰기

	int all_pair_img_cnt = 0;
	//for (int i = ncam-1; i >= 1; --i)
	for (int i = 0; i < ncam - 1; ++i)
	{
		//string img_path = "F:/_RGBD_Dataset7_Scenes/" + dataname + "jpgFiles/" + names[i];
		string img_path = "F:/_ChungAngUniv/" + dataname + "ALL/" + names[i];

		cout << "ref id: " << i << ", path:" << names[i] << endl;


		int ref_label_size = vec_db_image[i].voxel_db_pt.size();
		int ref_2dpts_sz = 0;
		for (int ix = 0; ix < ref_label_size; ++ix)
		{
			ref_2dpts_sz += vec_db_image[i].voxel_db_pt[ix].size();
		}

		int pair_cnt = 0;
		//for (int tar_im_id = i - 1; tar_im_id >= 0; --tar_im_id)
		for (int tar_im_id = i + 1; tar_im_id < ncam; ++tar_im_id)
		{
			if (pair_cnt >= 5)  // pair쌍 너무 많으면 다음 레퍼런스... 너무 많으니깐 ^^ -> 7쌍까지만...
				break;

			string img_path2 = "F:/_ChungAngUniv/" + dataname + "ALL/" + names[tar_im_id];
			//cout << "tar: " << names[tar_im_id] << endl;

			vector<string> line_str = split(names[i], '.');
			vector<string> img_path_split = split(line_str[0], '_');

			vector<string> line_str2 = split(names[tar_im_id], '.');
			vector<string> img_path_split2 = split(line_str2[0], '_');

			if (img_path_split[0] != img_path_split2[0])
			{
				cout << img_path_split[0] << " != " << img_path_split2[0] << endl;
				cout << "이미지 경로가 다르다~! ^^ skip 하자!" << endl;
				continue;
			}

			//Mat img = imread(img_path);
			//Mat down_img = img.clone();
			//resize(down_img, down_img, Size(reSize_W, reSize_H));  
			//Mat img2 = imread(img_path2);
			//Mat down_img2 = img2.clone();
			//resize(down_img2, down_img2, Size(reSize_W, reSize_H));  

			int tar_label_size = vec_db_image[tar_im_id].voxel_db_pt.size();
			int tar_2dpts_sz = 0;
			for (int ix = 0; ix < ref_label_size; ++ix)
			{
				tar_2dpts_sz += vec_db_image[tar_im_id].voxel_db_pt[ix].size();
			}

			int same_pt_cnt = 0;
			DB_Image2 ref_tmp_dbImg, tar_tmp_dbImg;
			vector<vector<DB_Point>> ref_tmp_dbPt, tar_tmp_dbPt;
			vector<vector<DB_Point>> ref_label_tmp_dbPt, tar_label_tmp_dbPt;
			ref_label_tmp_dbPt.resize(nLabel);
			tar_label_tmp_dbPt.resize(nLabel);

			//ref와 tar의 voxel별 포인트를 검색하기 위한 루프.
			for (int ref_lb_id = 0; ref_lb_id < ref_label_size; ++ref_lb_id)
			{
				int ref_lb_pt_sz = vec_db_image[i].voxel_db_pt[ref_lb_id].size();
				if (ref_lb_pt_sz < 3) continue;

				for (int tar_lb_id = 0; tar_lb_id < tar_label_size; ++tar_lb_id)
				{
					int tar_lb_pt_sz = vec_db_image[tar_im_id].voxel_db_pt[tar_lb_id].size();
					if (tar_lb_pt_sz < 3) continue;
					if (ref_lb_id == tar_lb_id)
					{
						//ref와 tar의 라벨 ID가 같을 때에만, 포인트 검색하여, 같은 3d_id를 갖는 pt를 찾고, 저장
						for (int ref_pt_id = 0; ref_pt_id < ref_lb_pt_sz; ++ref_pt_id)
						{
							int ref_pt_3d_id = vec_db_image[i].voxel_db_pt[ref_lb_id][ref_pt_id].pt_3d_id;
							for (int tar_pt_id = 0; tar_pt_id < tar_lb_pt_sz; ++tar_pt_id)
							{
								int tar_pt_3d_id = vec_db_image[tar_im_id].voxel_db_pt[tar_lb_id][tar_pt_id].pt_3d_id;

								if (ref_pt_3d_id == tar_pt_3d_id)        //같은게 나오면 그 다음은 없기 때문에,  target의 루프를 탈출 한다.
								{
									++same_pt_cnt;

									int ref_lb = vec_db_image[i].voxel_db_pt[ref_lb_id][ref_pt_id].clusterID - 1;
									ref_label_tmp_dbPt[ref_lb].push_back(vec_db_image[i].voxel_db_pt[ref_lb_id][ref_pt_id]);

									int tar_lb = vec_db_image[tar_im_id].voxel_db_pt[tar_lb_id][tar_pt_id].clusterID - 1;
									tar_label_tmp_dbPt[tar_lb].push_back(vec_db_image[tar_im_id].voxel_db_pt[tar_lb_id][tar_pt_id]);

									int feature_x = vec_db_image[i].voxel_db_pt[ref_lb_id][ref_pt_id].x_2d;
									int feature_y = vec_db_image[i].voxel_db_pt[ref_lb_id][ref_pt_id].y_2d;

									int feature_x2 = vec_db_image[tar_im_id].voxel_db_pt[tar_lb_id][tar_pt_id].x_2d;
									int feature_y2 = vec_db_image[tar_im_id].voxel_db_pt[tar_lb_id][tar_pt_id].y_2d;

									//circle(down_img, Point(feature_x, feature_y), 1, Scalar(255, 0, 0), -1);
									//circle(down_img2, Point(feature_x2, feature_y2), 1, Scalar(255, 0, 0), -1);

									break;   //같은게 나오면 그 다음은 없기 때문에,  target의 루프를 탈출 한다.
								}
							}
						}

					}

				}

			}

			int min_pt_cnt_thresh = MIN(ref_2dpts_sz, tar_2dpts_sz) * 0.4;

			//if (same_pt_cnt > corresp_threshold && same_pt_cnt >= 50)  //특정 개수(최소 50개는 있자)와 비율로.. 이상일 때에만 이미지 페어 저장, DB 저장.
			if (same_pt_cnt >= min_pt_cnt_thresh && same_pt_cnt >= 80)  //특정 개수(최소 70개는 있자) 이상일 때에만 이미지 페어 저장, DB 저장.
			{
				printf("i:%d, tar_im_id:%d, same_pt_cnt:%d, min_pt_cnt_thresh: %d\n", i, tar_im_id, same_pt_cnt, min_pt_cnt_thresh);


				++all_pair_img_cnt;
				++pair_cnt;


				Mat img2 = imread(img_path2);
				Mat down_img2 = img2.clone();
				resize(down_img2, down_img2, Size(reSize_W, reSize_H));  //455, 256

				Mat img = imread(img_path);
				Mat down_img = img.clone();
				resize(down_img, down_img, Size(reSize_W, reSize_H));  //455, 256

				Mat kp_label_img = Mat::zeros(reSize_H, reSize_W, CV_8U);
				Mat kp_label_img2 = Mat::zeros(reSize_H, reSize_W, CV_8U);

				Mat ref_label_img = Mat::zeros(down_img.rows, down_img.cols, CV_16U);
				Mat tar_label_img = Mat::zeros(down_img.rows, down_img.cols, CV_16U);
				//Mat ref_label_img = Mat::zeros(down_img.rows, down_img.cols, CV_8U);
				//Mat tar_label_img = Mat::zeros(down_img.rows, down_img.cols, CV_8U);
				for (int j = 0; j < ref_label_tmp_dbPt.size(); ++j)
				{
					if (ref_label_tmp_dbPt[j].size() < 3)   //이 부분이 voxel labeling convexhull 잡아주는 개수 부분... 1말고... 딴걸루 하장...
						continue;
					vector<Point> ref_contour;

					for (int k = 0; k < ref_label_tmp_dbPt[j].size(); ++k)
					{
						int feature_x = ref_label_tmp_dbPt[j][k].x_2d;
						int feature_y = ref_label_tmp_dbPt[j][k].y_2d;

						kp_label_img.data[(int)feature_y*kp_label_img.cols + (int)feature_x] = 255;
						ref_contour.push_back(Point(feature_x, feature_y));
					}
					ref_tmp_dbPt.push_back(ref_label_tmp_dbPt[j]);

					vector<Point> hull(ref_contour.size());
					convexHull(ref_contour, hull);
					const Point *pts = (const Point*)Mat(hull).data;
					int npts = Mat(hull).rows;

					ushort val = ref_label_tmp_dbPt[j][0].clusterID; //배경이 0이라고 하면, 그냥 clusterID를 넘겨주면 됨. 
					fillPoly(ref_label_img, &pts, &npts, 1, Scalar(val));

				}

				for (int j = 0; j < tar_label_tmp_dbPt.size(); ++j)
				{
					if (tar_label_tmp_dbPt[j].size() < 3) //이 부분이 target voxel labeling convexhull 잡아주는 개수 부분... 1말고... 딴걸루 하장...
						continue;
					vector<Point> tar_contour;
					for (int k = 0; k < tar_label_tmp_dbPt[j].size(); ++k)
					{
						int feature_x = tar_label_tmp_dbPt[j][k].x_2d;
						int feature_y = tar_label_tmp_dbPt[j][k].y_2d;

						kp_label_img2.data[(int)feature_y*kp_label_img2.cols + (int)feature_x] = 255;

						tar_contour.push_back(Point(feature_x, feature_y));
					}
					tar_tmp_dbPt.push_back(tar_label_tmp_dbPt[j]);

					vector<Point> hull(tar_contour.size());
					convexHull(tar_contour, hull);
					const Point *pts = (const Point*)Mat(hull).data;
					int npts = Mat(hull).rows;

					ushort val = tar_label_tmp_dbPt[j][0].clusterID; //배경이 0이라고 하면, 그냥 clusterID를 넘겨주면 됨. 
					fillPoly(tar_label_img, &pts, &npts, 1, Scalar(val));

				}

				//imshow("ref ", down_img);
				//imshow("tar ", down_img2);
				//imshow("ref kp", kp_label_img);
				//imshow("tar kp", kp_label_img2);
				//imshow("ref_label_img", ref_label_img);
				//imshow("tar_label_img", tar_label_img);
				//waitKey();

				//vector<string> line_str = split(names[i], '.');
				//string ref_file_name =  to_string(i) + '_' + to_string(tar_im_id) + '_' + "ref_" + line_str[0] + ".jpg";
				//ref_file_name.replace(ref_file_name.find("jpg"), 3, "png");

				//line_str = split(names[tar_im_id], '.');
				//string tar_file_name = to_string(i) + '_' + to_string(tar_im_id) + '_' + "tar_" + line_str[0] + ".jpg";
				//tar_file_name.replace(tar_file_name.find("jpg"), 3, "png");


				//all 일때만... 적용하자.
				vector<string> line_str = split(names[i], '.');
				vector<string> img_path_split = split(line_str[0], '_');
				string ref_file_name = img_path_split[0] + "/" + to_string(i) + '_' + to_string(tar_im_id) + '_' 
					+ "ref_" + img_path_split[1] + img_path_split[2] + ".png";

				line_str = split(names[tar_im_id], '.');
				img_path_split = split(line_str[0], '_');
				string tar_file_name = img_path_split[0] + "/" + to_string(i) + '_' + to_string(tar_im_id) + '_' 
					+ "tar_" + img_path_split[1] + img_path_split[2] + ".png";


				string save_img_path = "F:/_voxelFeatureMap_ChungAngUni_Scenes/pair_Voxel_set/" + dataname + ref_file_name;
				imwrite(save_img_path, down_img);
				save_img_path.replace(save_img_path.find("ref_"), 4, "ref_label_");
				imwrite(save_img_path, ref_label_img);
				save_img_path.replace(save_img_path.find("label"), 5, "kp");
				imwrite(save_img_path, kp_label_img);

				string save_img_path2 = "F:/_voxelFeatureMap_ChungAngUni_Scenes/pair_Voxel_set/" + dataname + tar_file_name;
				imwrite(save_img_path2, down_img2);
				save_img_path2.replace(save_img_path2.find("tar_"), 4, "tar_label_");
				imwrite(save_img_path2, tar_label_img);
				save_img_path2.replace(save_img_path2.find("label"), 5, "kp");
				imwrite(save_img_path2, kp_label_img2);


				pair<DB_Image2, DB_Image2> pair_DB_img;

				ref_tmp_dbImg = vec_db_image[i];
				ref_tmp_dbImg.voxel_db_pt = ref_tmp_dbPt;
				ref_tmp_dbImg.img_path = ref_file_name;
				tar_tmp_dbImg = vec_db_image[tar_im_id];
				tar_tmp_dbImg.voxel_db_pt = tar_tmp_dbPt;
				tar_tmp_dbImg.img_path = tar_file_name;

				pair_DB_img = make_pair(ref_tmp_dbImg, tar_tmp_dbImg);


				//text파일로 저장하는 방법 - pt_2d_id는 저장 안함.
				oStream_first << pair_DB_img.first.img_ID << " " << pair_DB_img.first.img_path << " " << pair_DB_img.first.quat[0] << " " << pair_DB_img.first.quat[1]
					<< " " << pair_DB_img.first.quat[2] << " " << pair_DB_img.first.quat[3] << " " << pair_DB_img.first.camCent[0] << " " << pair_DB_img.first.camCent[1]
					<< " " << pair_DB_img.first.camCent[2] << " " << pair_DB_img.first.focal_lenX << " " << pair_DB_img.first.focal_lenY
					<< " " << pair_DB_img.first.Cx << " " << pair_DB_img.first.Cy << " ";

				int ref_pt_size_sum = 0;
				for (int lb_id = 0; lb_id < pair_DB_img.first.voxel_db_pt.size(); ++lb_id)
				{
					int lb_pt_sz = pair_DB_img.first.voxel_db_pt[lb_id].size();
					ref_pt_size_sum += lb_pt_sz;
				}
				oStream_first << ref_pt_size_sum << endl;

				for (int lb_id = 0; lb_id < pair_DB_img.first.voxel_db_pt.size(); ++lb_id)
				{
					int lb_pt_sz = pair_DB_img.first.voxel_db_pt[lb_id].size();
					for (int pt_id = 0; pt_id < lb_pt_sz; ++pt_id)
					{
						oStream_first << pair_DB_img.first.voxel_db_pt[lb_id][pt_id].x_2d << " " << pair_DB_img.first.voxel_db_pt[lb_id][pt_id].y_2d
							<< " " << pair_DB_img.first.voxel_db_pt[lb_id][pt_id].x << " " << pair_DB_img.first.voxel_db_pt[lb_id][pt_id].y << " " << pair_DB_img.first.voxel_db_pt[lb_id][pt_id].z
							<< " " << pair_DB_img.first.voxel_db_pt[lb_id][pt_id].clusterID << " " << pair_DB_img.first.voxel_db_pt[lb_id][pt_id].pt_3d_id << endl;
					}

				}

				oStream_second << pair_DB_img.second.img_ID << " " << pair_DB_img.second.img_path << " " << pair_DB_img.second.quat[0] << " " << pair_DB_img.second.quat[1]
					<< " " << pair_DB_img.second.quat[2] << " " << pair_DB_img.second.quat[3] << " " << pair_DB_img.second.camCent[0] << " " << pair_DB_img.second.camCent[1]
					<< " " << pair_DB_img.second.camCent[2] << " " << pair_DB_img.second.focal_lenX << " " << pair_DB_img.second.focal_lenY
					<< " " << pair_DB_img.second.Cx << " " << pair_DB_img.second.Cy << " ";



				int tar_pt_size_sum = 0;
				for (int lb_id = 0; lb_id < pair_DB_img.second.voxel_db_pt.size(); ++lb_id)
				{
					int lb_pt_sz = pair_DB_img.second.voxel_db_pt[lb_id].size();
					tar_pt_size_sum += lb_pt_sz;
				}
				oStream_second << tar_pt_size_sum << endl;

				for (int lb_id = 0; lb_id < pair_DB_img.second.voxel_db_pt.size(); ++lb_id)
				{
					int lb_pt_sz = pair_DB_img.second.voxel_db_pt[lb_id].size();
					for (int pt_id = 0; pt_id < lb_pt_sz; ++pt_id)
					{
						oStream_second << pair_DB_img.second.voxel_db_pt[lb_id][pt_id].x_2d << " " << pair_DB_img.second.voxel_db_pt[lb_id][pt_id].y_2d
							<< " " << pair_DB_img.second.voxel_db_pt[lb_id][pt_id].x << " " << pair_DB_img.second.voxel_db_pt[lb_id][pt_id].y << " " << pair_DB_img.second.voxel_db_pt[lb_id][pt_id].z
							<< " " << pair_DB_img.second.voxel_db_pt[lb_id][pt_id].clusterID << " " << pair_DB_img.second.voxel_db_pt[lb_id][pt_id].pt_3d_id << endl;
					}
				}



				//imshow("img1", down_img);
				//imshow("img2", down_img2);
				//imshow("kp_label_img1", kp_label_img);
				//imshow("kp_label_img2", kp_label_img2);
				//waitKey();
			}
			else
			{
				//printf("대응점 개수가 임계치보다 낮습니다...\n");
				//printf("corresp_threshold:%d\n", min_pt_cnt_thresh);
				//printf("same_pt_cnt:%d\n", same_pt_cnt);
				continue;
			}

			//imshow("img1", down_img);
			//imshow("img2", down_img2);
			//imshow("kp_label_img1", kp_label_img);
			//imshow("kp_label_img2", kp_label_img2);
			//waitKey();

		}  //for (int tar_im_id = i+1; tar_im_id < ncam; ++tar_im_id)

		printf("ref id:%d, pair_cnt:%d\n", i, pair_cnt);

	}
	oStream_first.close();   // 파일 닫기
	oStream_second.close();
	printf("all_pair_img_cnt:%d\n", all_pair_img_cnt);

	printf("finish!!!\n");

	return true;
}


void SaveNVM(const char* filename, vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
	vector<string>& names, vector<int>& ptc)
{
	std::cout << "Saving model to " << filename << "...\n";
	ofstream out(filename);

	out << "NVM_V3_R9T\n" << camera_data.size() << '\n' << std::setprecision(12);
	if (names.size() < camera_data.size()) names.resize(camera_data.size(), string("unknown"));
	if (ptc.size() < 3 * point_data.size()) ptc.resize(point_data.size() * 3, 0);

	////////////////////////////////////
	for (size_t i = 0; i < camera_data.size(); ++i)
	{
		CameraT& cam = camera_data[i];
		out << names[i] << ' ' << cam.GetFocalLength() << ' ';
		for (int j = 0; j < 9; ++j) out << cam.m[0][j] << ' ';
		out << cam.t[0] << ' ' << cam.t[1] << ' ' << cam.t[2] << ' '
			<< cam.GetNormalizedMeasurementDistortion() << " 0\n";
	}

	out << point_data.size() << '\n';

	for (size_t i = 0, j = 0; i < point_data.size(); ++i)
	{
		Point3D& pt = point_data[i];
		int * pc = &ptc[i * 3];
		out << pt.xyz[0] << ' ' << pt.xyz[1] << ' ' << pt.xyz[2] << ' '
			<< pc[0] << ' ' << pc[1] << ' ' << pc[2] << ' ';

		size_t je = j;
		while (je < ptidx.size() && ptidx[je] == (int)i) je++;

		out << (je - j) << ' ';

		for (; j < je; ++j)    out << camidx[j] << ' ' << " 0 " << measurements[j].x << ' ' << measurements[j].y << ' ';

		out << '\n';
	}
}



bool LoadModelFile(string& name, string& dataname, vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
	vector<string>& names, vector<int>& ptc, int cubeSize)
{
	if (name == "")return false;
	ifstream in(name);

	std::cout << "Loading cameras/points: " << name << "\n";
	if (!in.is_open()) return false;

	return LoadNVM(in, dataname, camera_data, point_data, measurements, ptidx, camidx, names, ptc, cubeSize);
}


float random_ratio(float percent)
{
	return (rand() % 101 - 50) * 0.02f * percent + 1.0f;
}


bool RemoveInvisiblePoints(vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<int>& ptidx, vector<int>& camidx,
	vector<Point2D>& measurements, vector<string>& names, vector<int>& ptc)
{
	vector<float> zz(ptidx.size());
	for (size_t i = 0; i < ptidx.size(); ++i)
	{
		CameraT& cam = camera_data[camidx[i]];
		Point3D& pt = point_data[ptidx[i]];
		zz[i] = cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] + cam.m[2][2] * pt.xyz[2] + cam.t[2];
	}
	size_t median_idx = ptidx.size() / 2;
	std::nth_element(zz.begin(), zz.begin() + median_idx, zz.end());
	float dist_threshold = zz[median_idx] * 0.001f;

	//keep removing 3D points. until all of them are infront of the cameras..
	vector<bool> pmask(point_data.size(), true);
	int points_removed = 0;
	for (size_t i = 0; i < ptidx.size(); ++i)
	{
		int cid = camidx[i], pid = ptidx[i];
		if (!pmask[pid])continue;
		CameraT& cam = camera_data[cid];
		Point3D& pt = point_data[pid];
		bool visible = (cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] + cam.m[2][2] * pt.xyz[2] + cam.t[2] > dist_threshold);
		pmask[pid] = visible; //this point should be removed
		if (!visible) points_removed++;
	}
	if (points_removed == 0) return false;
	vector<int>  cv(camera_data.size(), 0);
	//should any cameras be removed ?
	int min_observation = 20; //cameras should see at leat 20 points

	do
	{
		//count visible points for each camera
		std::fill(cv.begin(), cv.end(), 0);
		for (size_t i = 0; i < ptidx.size(); ++i)
		{
			int cid = camidx[i], pid = ptidx[i];
			if (pmask[pid])  cv[cid]++;
		}

		//check if any more points should be removed
		vector<int>  pv(point_data.size(), 0);
		for (size_t i = 0; i < ptidx.size(); ++i)
		{
			int cid = camidx[i], pid = ptidx[i];
			if (!pmask[pid]) continue; //point already removed
			if (cv[cid] < min_observation) //this camera shall be removed.
			{
				///
			}
			else
			{
				pv[pid]++;
			}
		}

		points_removed = 0;
		for (size_t i = 0; i < point_data.size(); ++i)
		{
			if (pmask[i] == false) continue;
			if (pv[i] >= 2) continue;
			pmask[i] = false;
			points_removed++;
		}
	} while (points_removed > 0);

	////////////////////////////////////
	vector<bool> cmask(camera_data.size(), true);
	for (size_t i = 0; i < camera_data.size(); ++i) cmask[i] = cv[i] >= min_observation;
	////////////////////////////////////////////////////////

	vector<int> cidx(camera_data.size());
	vector<int> pidx(point_data.size());




	///modified model.
	vector<CameraT> camera_data2;
	vector<Point3D> point_data2;
	vector<int> ptidx2;
	vector<int> camidx2;
	vector<Point2D> measurements2;
	vector<string> names2;
	vector<int> ptc2;


	//
	if (names.size() < camera_data.size()) names.resize(camera_data.size(), string("unknown"));
	if (ptc.size() < 3 * point_data.size()) ptc.resize(point_data.size() * 3, 0);

	//////////////////////////////
	int new_camera_count = 0, new_point_count = 0;
	for (size_t i = 0; i < camera_data.size(); ++i)
	{
		if (!cmask[i])continue;
		camera_data2.push_back(camera_data[i]);
		names2.push_back(names[i]);
		cidx[i] = new_camera_count++;
	}

	for (size_t i = 0; i < point_data.size(); ++i)
	{
		if (!pmask[i])continue;
		point_data2.push_back(point_data[i]);
		ptc.push_back(ptc[i]);
		pidx[i] = new_point_count++;
	}

	int new_observation_count = 0;
	for (size_t i = 0; i < ptidx.size(); ++i)
	{
		int pid = ptidx[i], cid = camidx[i];
		if (!pmask[pid] || !cmask[cid]) continue;
		ptidx2.push_back(pidx[pid]);
		camidx2.push_back(cidx[cid]);
		measurements2.push_back(measurements[i]);
		new_observation_count++;
	}

	std::cout << "NOTE: removing " << (camera_data.size() - new_camera_count) << " cameras; " << (point_data.size() - new_point_count)
		<< " 3D Points; " << (measurements.size() - new_observation_count) << " Observations;\n";

	camera_data2.swap(camera_data); names2.swap(names);
	point_data2.swap(point_data);   ptc2.swap(ptc);
	ptidx2.swap(ptidx);  camidx2.swap(camidx);
	measurements2.swap(measurements);

	return true;
}
