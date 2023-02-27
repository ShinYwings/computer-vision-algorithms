#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class scanning {

	private:

		const int threadnum;
		int image_width;
		int image_height;
		int* startingPoint_each_piece;
		uchar** original_image;
		uchar*** pieces;
		const int threshold;
		
	public:

		void getImage(Mat newimage);
		void divideImage();
		vector<pair<int, int>> makeIter(int threshold, int imageHeight, int threadnum, int bias);
		int* getStartingPoint_each_piece() const { return this->startingPoint_each_piece;};
		
		pair<int, uchar*> debug();
		uchar*** imageOut() { return this->pieces; }

		scanning(int threshold, int threadnum) :threshold(threshold),image_width(0), image_height(0), 
									  threadnum(threadnum),original_image(nullptr), pieces(nullptr),
										startingPoint_each_piece(nullptr){};

		~scanning() {
			cout << "destructor called" << endl;

			// 이렇게 하면 배열 아래에 있는 것들이 여전히 메모리 할당되어있음...
			// for문 써가지고 배열 차원 만큼 delete 해줘야하는데 귀찮..
			delete original_image;
			delete pieces;
			delete startingPoint_each_piece;
		};
};

