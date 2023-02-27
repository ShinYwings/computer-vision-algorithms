#include "scanning.h"

void scanning::getImage(Mat image) {

	this->image_width = image.cols;
	this->image_height = image.rows;
 	this->original_image = new uchar*[image.rows];

	for (int i = 0; i < this->image_height; i++) {
		this->original_image[i] = new uchar[image.cols];
	}
	
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			this->original_image [i][j] = image.at<uchar>(i,j);  // row, col
		}
	}
}

void scanning::divideImage() {

	int thread_num = this->threadnum;
	int interval = this->image_height / thread_num;
	int threshold = this->threshold;
	
	int bias {0};
	if (interval * thread_num != this->image_height) {
		bias = this->image_height - (interval * thread_num);
	}

	vector<pair<int, int>> iter = makeIter(threshold, image_height, thread_num, bias);

	this->startingPoint_each_piece = new int[thread_num];
	for (int idx = 0; idx < thread_num; idx++) {
		this->startingPoint_each_piece[idx] = iter[idx].first;
	}

	this->pieces = new uchar **[thread_num];
	for (int index = 0; index < thread_num; index++) {

		int start = iter[index].first;
		int end = iter[index].second;
		this->pieces[index] = new uchar *[(end-start)];

		for (int i = 0; i < end-start; i++) {
			this->pieces[index][i] = new uchar[this->image_width];
			for (int j = 0; j < this->image_width; j++) {
					this->pieces[index][i][j] = this->original_image[i+start][j];
			}
		}
	}
	cout << "successfully divided into "<< thread_num <<" pieces" << endl;
}

pair<int, uchar*> scanning::debug() {

	int imageHeight = this->image_height;
	int threadnum = this->threadnum;
	int pieceHeight = imageHeight / threadnum + (this->threshold*2);

	uchar* real = new uchar[((threadnum * imageHeight) * pieceHeight)];

	for (int index = 0; index < threadnum; index++) {

		for (int q = 0; q < pieceHeight; q++)
		{
			for (int t = 0; t < imageHeight; t++)
			{
				real[((index * pieceHeight) + q) * imageHeight + t] = this->pieces[index][q][t];
			}
		}
	}
	pair<int, uchar*> result = { pieceHeight, real };
	return result;
}

vector<pair<int, int>> scanning::makeIter(int threshold, int imageHeight, int threadnum, int bias) {

	int interval = imageHeight / threadnum;
	vector<pair<int, int>> miter;

	for (int i = 0; i < threadnum; i++) {

		if (i == 0) {
			miter.push_back({ 0, interval + threshold * 2 });
		}
		else if (i == threadnum - 1) {
			miter.push_back({ interval * i - threshold * (i - 1) - bias, imageHeight });
		}
		else {
			miter.push_back({ interval * i - threshold, interval * (i + 1) + threshold });
		}
	}

	return miter;
}