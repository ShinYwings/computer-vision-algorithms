#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int NUM_EIGEN_FACES = 8;
const int MAX_SLIDER_VALUE = 255;

class preprocessing {

private:
	vector<Mat> images;
	pair<int, int> img_h_w;
	
	void svd(std::vector<std::vector<float>> matrix, std::vector<std::vector<float>>& s,
		std::vector<std::vector<float>>& u, std::vector<std::vector<float>>& v);
	
	void compute_evd(std::vector<std::vector<float>> matrix,
		std::vector<float>& eigenvalues, std::vector<std::vector<float>>& eigenvectors, std::size_t eig_count);
	void get_hermitian_matrix(std::vector<float> eigenvector, std::vector<std::vector<float>>& h_matrix);
	void get_hermitian_matrix_inverse(std::vector<float> eigenvector, std::vector<std::vector<float>>& ih_matrix);
	void get_inverse_diagonal_matrix(std::vector<std::vector<float>> matrix, std::vector<std::vector<float>>& inv_matrix);
	void get_reduced_matrix(std::vector<std::vector<float>> matrix, std::vector<std::vector<float>>& r_matrix, std::size_t new_size);
	void matrix_by_matrix(std::vector<std::vector<float>> matrix1, std::vector<std::vector<float>>& matrix2, std::vector<std::vector<float>>& matrix3);
	void matrix_transpose(std::vector<std::vector<float>> matrix1, std::vector<std::vector<float>>& matrix2);
	void jordan_gaussian_transform(std::vector<std::vector<float>> matrix, std::vector<float>& eigenvector);

public:
	pair<vector<vector<float>>, vector<float>> createDataMatrix(string* dirName);
	Mat PCA(pair<vector<vector<float>>, vector<float>> data_args);

	preprocessing() :images() {};
	~preprocessing() {
		std::cout << "preprocessing destructor call " << std::endl;
	};


};