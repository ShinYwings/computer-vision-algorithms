#include <opencv4/opencv2/opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;

#define PI 3.14159265
#define MAX 99999

void onMouse_first(int event, int x, int y, int flags, void* param);
void onMouse_second(int event, int x, int y, int flags, void* param);
void sobelFunc(Mat image1, Mat image2);
float sub(float h1[8], float h2[8]);
int findMin(float saving[4]);

int pointers_1[4][2];
int pointers_2[4][2];
int counter_1 = 0;
int counter_2 = 0;

Mat first;
Mat second;

bool activate = false;

int main() {

	first = imread("1st.jpg", IMREAD_GRAYSCALE);
	second = imread("2nd.jpg", IMREAD_GRAYSCALE);

	if (first.empty())
		return -1;

	if (second.empty())
		return -1;

	resize(first, first, Size(first.cols/8, first.rows/8));
	resize(second, second, Size(second.cols /8, second.rows /8));

	imshow("first", first);
	setMouseCallback("first", onMouse_first, (void*)&first);

	imshow("second", second);
	setMouseCallback("second", onMouse_second, (void*)&second);

	waitKey();

	return 0;
}

void onMouse_first(int event, int x, int y, int flags, void* param) {
	Mat *pMat = (Mat*)param;
	Mat image = Mat(*pMat);

	int ksize = 15;

	if (event == EVENT_LBUTTONDOWN && counter_1 < 4) {
		rectangle(image, Point(x - float(ksize) / 2.0, y - float(ksize) / 2.0), Point(x + float(ksize) / 2.0, y + float(ksize) / 2.0), Scalar(0, 0, 255));

		pointers_1[counter_1][0] = x;
		pointers_1[counter_1][1] = y;

		counter_1 += 1;

		imshow("first", image);

		for (int i = 0; i < 4; i++) {
			cout << i << " x1:  " << pointers_1[i][0] << " y1:  " << pointers_1[i][1] << '\n';
		}
		cout << '\n';
	}
}

void onMouse_second(int event, int x, int y, int flags, void* param) {
	Mat *pMat = (Mat*)param;
	Mat image = Mat(*pMat);

	int ksize = 15;

	if (event == EVENT_LBUTTONDOWN && counter_2 < 4) {
		
		rectangle(image, Point(x - float(ksize) / 2.0, y - float(ksize) / 2.0), Point(x + float(ksize) / 2.0, y + float(ksize) / 2.0), Scalar(0, 0, 255));

		pointers_2[counter_2][0] = x;
		pointers_2[counter_2][1] = y;

		counter_2 += 1;

		imshow("second", image);

		for (int i = 0; i < 4; i++) {
			cout << i << " x2:  " << pointers_2[i][0] << " y2:  " << pointers_2[i][1] << '\n';
		}
		cout << '\n';

		if (counter_1 == 4 && counter_2 == 4) {
			activate = true;
			sobelFunc(first, second);
		}
	}
}

void sobelFunc(Mat first, Mat second) {
	int ksize = 15;

	Mat cropImg_1[4];
	Mat dstMag_1[4];

	Mat cropImg_2[4];
	Mat dstMag_2[4];

	float ori_1[15][15] = { 0, };
	float ori_2[15][15] = { 0, };

	float hist1[4][8] = { 0, };
	float hist2[4][8] = { 0, };

	if (activate){
		for (int i = 0; i < 4; i++) {

			Rect boundary(0, 0, first.cols, first.rows);
			Rect roi(pointers_1[i][0] - 7, pointers_1[i][1] - 7, 15, 15);
			cropImg_1[i] = first(roi & boundary);

			int ddepth = CV_32F;
			Mat dstGx, dstGy;
			Sobel(cropImg_1[i], dstGx, ddepth, 1, 0, ksize);
			Sobel(cropImg_1[i], dstGy, ddepth, 0, 1, ksize);

			magnitude(dstGx, dstGy, dstMag_1[i]);
			for (int j = 0; j < 15; j++) {
				for (int k = 0; k < 15; k++) {
					float orientation = atan2(dstGx.at<float>(j, k), dstGy.at<float>(j, k)) * 180 / PI;
					if (orientation < 0)
						orientation = 360 + orientation;
					ori_1[j][k] = orientation;

					int index = int(ori_1[j][k]) % 360 / 45.;
					hist1[i][index] += (1e-9*dstMag_1[i].at<float>(j, k));
				}
			}
		}

		for (int i = 0; i < 4; i++) {

			Rect boundary(0, 0, second.cols, second.rows);
			Rect roi(pointers_2[i][0] - 7, pointers_2[i][1] - 7, 15, 15);
			cropImg_2[i] = second(roi & boundary);

			int ddepth = CV_32F;
			Mat dstGx, dstGy;
			Sobel(cropImg_2[i], dstGx, ddepth, 1, 0, ksize);
			Sobel(cropImg_2[i], dstGy, ddepth, 0, 1, ksize);

			magnitude(dstGx, dstGy, dstMag_2[i]);
			for (int j = 0; j < 15; j++) {
				for (int k = 0; k < 15; k++) {
					float orientation = atan2(dstGx.at<float>(j, k), dstGy.at<float>(j, k)) * 180 / PI;
					if (orientation < 0)
						orientation = 360 + orientation;
					ori_2[j][k] = orientation;
					int index = int(int(ori_2[j][k]) % 360 / 45.);
					hist2[i][index] += (1e-9*dstMag_2[i].at<float>(j, k));
				}
			}
		}
		activate = false;
		

		// 대표 방향에 대해 normalize
		int max_idx;
		float max_val;

		for (int i = 0; i < 4; i++) {
			max_idx = -1;
			max_val = 0;

			for (int j = 0; j < 8; j++) {
				// find max index (대표 방향성)
				if (hist1[i][j] > max_val) {
					max_val = hist1[i][j];
					max_idx = j;
				}
			}
			// 방향성 통일
			float temp[8] = { 0, };
			for (int k = 0; k < 8; k++)
				temp[k] = hist1[i][(max_idx + k) % 8];
			for (int k = 0; k < 8; k++)
				hist1[i][k] = temp[k];
		}
		
		for (int i = 0; i < 4; i++) {
			max_idx = -1;
			max_val = 0;
			for (int j = 0; j < 8; j++) {
				// find max index (대표 방향성)
				if (hist2[i][j] > max_val) {
					max_val = hist2[i][j];
					max_idx = j;
				}
			}

			// 방향성 통일
			float temp[8] = { 0, };
			for (int k = 0; k < 8; k++)
				temp[k] = hist2[i][(max_idx + k) % 8];

			for (int k = 0; k < 8; k++)
				hist2[i][k] = temp[k];
		}

		// histogram
		Mat histImage(256, 256, CV_8U);
		histImage = Scalar(255);
		int histSize = 8;
		int binW = cvRound((double)histImage.cols / histSize);
		string name[8] = { "h11", "h12", "h13","h14",
						   "h21", "h22", "h23","h24" };

		for (int c = 0; c < 4; c++) {
			for (int k = 0; k < 8; k++) {
				int x1 = k * binW;
				int y1 = histImage.rows;
				int x2 = (k + 1) * binW;
				int y2 = histImage.rows - cvRound(hist1[c][k]);

				rectangle(histImage, Point(x1, y1), Point(x2, y2), Scalar(0));
			}
			imshow(name[c], histImage);
			histImage = Scalar(255);
			waitKey();
		}

		for (int c = 0; c < 4; c++) {
			for (int k = 0; k < 8; k++) {
				int x1 = k * binW;
				int y1 = histImage.rows;
				int x2 = (k + 1) * binW;
				int y2 = histImage.rows - cvRound(hist2[c][k]);

				rectangle(histImage, Point(x1, y1), Point(x2, y2), Scalar(0));
			}
			imshow(name[c+4], histImage);
			histImage = Scalar(255);
			waitKey();
		}

		
		int matching[4] = { -1,-1,-1,-1 };
		float saving[4] = { MAX, MAX, MAX, MAX };
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++)
				saving[j] = sub(hist1[i], hist2[j]);
			matching[i] = findMin(saving);
		}

		cout << "0: " << matching[0] << "  1:  " << matching[1] << "  2:  " << matching[2] << "  3:  " << matching[3] << '\n';

		// 이미지 그리기
		Mat dst(first.cols*2, 2*first.rows, CV_8U);
		hconcat(first, second, dst);
		for (int i = 0; i < 4; i++) {
			int idx = matching[i];
			line(dst, Point(pointers_1[i][0], pointers_1[i][1]), Point(pointers_2[idx][0] + first.cols, pointers_2[idx][1]), (255, 0, 0), 2);
		}
		imshow("final", dst);
		waitKey();
	} // if activate
	
	return;
}


float sub(float h1[8], float h2[8]) {
	float loss = 0;
	for (int i = 0; i < 8; i++)
		loss += abs(h1[i] - h2[i]);
	
	return loss / 8.;
}

int findMin(float saving[4]) {
	int idx = -1;
	float value = MAX;
	for (int i = 0; i < 4; i++) {
		if (value > saving[i]) {
			idx = i;
			value = saving[i];
		}
	}
	return idx;
}
