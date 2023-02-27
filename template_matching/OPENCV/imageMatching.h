#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <math.h>
#include <vector>
#include "PointInfo.h"

using namespace std;
using namespace cv;

const int INF = 9991000000;

class imageMatching {

	private:

		int threads_num;
		int image_width;
		int image_height;
		uchar** template_image;

	public:

		void getTemplateImage(Mat input_image);
		void worker(uchar** piece, pair<int, int> h_w, vector<PointInfo>* result);
		void SumofAbsoluteDifferences(uchar** data_original, pair<int, int> original_h_w, vector<PointInfo>* result);
		void ccoefficient_normed(uchar** data_original, pair<int, int> original_h_w, vector<PointInfo>* result);
		pair<int, vector<PointInfo>> matchingStart(uchar*** pieces, pair<int, int> h_w);
		int getImageHeight() const { return this->image_height; };
		int getImageWidth() const { return this->image_width; };

		
		imageMatching(int threads_num):image_width(0),image_height(0), threads_num(threads_num),
						 template_image(){};

		~imageMatching() {
			cout << "imageMatching destructor called" << endl;

			// �̷��� �ϸ� �迭 �Ʒ��� �ִ� �͵��� ������ �޸� �Ҵ�Ǿ�����...
			// for�� �ᰡ���� �迭 ���� ��ŭ delete ������ϴµ� ����..
			delete template_image;
		};
};
