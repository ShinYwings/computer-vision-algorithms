#include "preprocessing.h"

void preprocessing::svd(std::vector<std::vector<float>> matrix, std::vector<std::vector<float>>& s,
	std::vector<std::vector<float>>& u, std::vector<std::vector<float>>& v)
{
	std::vector<std::vector<float>> matrix_t;
	matrix_transpose(matrix, matrix_t);

	std::vector<std::vector<float>> matrix_product1;
	matrix_by_matrix(matrix, matrix_t, matrix_product1);

	std::vector<std::vector<float>> matrix_product2;
	matrix_by_matrix(matrix_t, matrix, matrix_product2);

	std::vector<std::vector<float>> u_1;
	std::vector<std::vector<float>> v_1;

	std::vector<float> eigenvalues;
	compute_evd(matrix_product2, eigenvalues, v_1, 0);

	matrix_transpose(v_1, v);

	s.resize(matrix.size());
	for (std::uint32_t index = 0; index < eigenvalues.size(); index++)
	{
		s[index].resize(eigenvalues.size());
		s[index][index] = eigenvalues[index];
	}

	std::vector<std::vector<float>> s_inverse;
	get_inverse_diagonal_matrix(s, s_inverse);

	std::vector<std::vector<float>> av_matrix;
	matrix_by_matrix(matrix, v, av_matrix);
	matrix_by_matrix(av_matrix, s_inverse, u);
}

void preprocessing::compute_evd(std::vector<std::vector<float>> matrix,
	std::vector<float>& eigenvalues, std::vector<std::vector<float>>& eigenvectors, std::size_t eig_count)
{
	std::size_t m_size = matrix.size();
	std::vector<float> vec; vec.resize(m_size);
	std::fill_n(vec.begin(), m_size, 1);

	static std::vector<std::vector<float>> matrix_i;

	if (eigenvalues.size() == 0 && eigenvectors.size() == 0)
	{
		eigenvalues.resize(m_size);
		eigenvectors.resize(eigenvalues.size());
		matrix_i = matrix;
	}

	std::vector<std::vector<float>> m; m.resize(m_size);
	for (std::uint32_t row = 0; row < m_size; row++)
		m[row].resize(100);

	float lambda_old = 0;

	std::uint32_t index = 0; bool is_eval = false;
	while (is_eval == false)
	{
		for (std::uint32_t row = 0; row < m_size && (index % 100) == 0; row++)
			m[row].resize(m[row].size() + 100);

		for (std::uint32_t row = 0; row < m_size; row++)
		{
			m[row][index] = 0;
			for (std::uint32_t col = 0; col < m_size; col++)
				m[row][index] += matrix[row][col] * vec[col];
		}

		for (std::uint32_t col = 0; col < m_size; col++)
			vec[col] = m[col][index];

		if (index > 0)
		{
			float lambda = (m[0][index - 1] != 0) ? \
				(m[0][index] / m[0][index - 1]) : m[0][index];
			is_eval = (std::fabs(lambda - lambda_old) < 0.0000000001) ? true : false;

			lambda = (std::fabs(lambda) >= 10e-6) ? lambda : 0;
			eigenvalues[eig_count] = lambda; lambda_old = lambda;
		}

		index++;
	}

	std::vector<std::vector<float>> matrix_new;

	if (m_size > 1)
	{
		std::vector<std::vector<float>> matrix_tfloatet;
		matrix_tfloatet.resize(m_size);

		for (std::uint32_t row = 0; row < m_size; row++)
		{
			matrix_tfloatet[row].resize(m_size);
			for (std::uint32_t col = 0; col < m_size; col++)
				matrix_tfloatet[row][col] = (row == col) ? \
				(matrix[row][col] - eigenvalues[eig_count]) : matrix[row][col];
		}

		std::vector<float> eigenvector;
		jordan_gaussian_transform(matrix_tfloatet, eigenvector);

		std::vector<std::vector<float>> hermitian_matrix;
		get_hermitian_matrix(eigenvector, hermitian_matrix);

		std::vector<std::vector<float>> ha_matrix_product;
		matrix_by_matrix(hermitian_matrix, matrix, ha_matrix_product);

		std::vector<std::vector<float>> inverse_hermitian_matrix;
		get_hermitian_matrix_inverse(eigenvector, inverse_hermitian_matrix);

		std::vector<std::vector<float>> iha_matrix_product;
		matrix_by_matrix(ha_matrix_product, inverse_hermitian_matrix, iha_matrix_product);

		get_reduced_matrix(iha_matrix_product, matrix_new, m_size - 1);
	}

	if (m_size <= 1)
	{
		for (std::uint32_t index = 0; index < eigenvalues.size(); index++)
		{
			float lambda = eigenvalues[index];
			std::vector<std::vector<float>> matrix_target;
			matrix_target.resize(matrix_i.size());

			for (std::uint32_t row = 0; row < matrix_i.size(); row++)
			{
				matrix_target[row].resize(matrix_i.size());
				for (std::uint32_t col = 0; col < matrix_i.size(); col++)
					matrix_target[row][col] = (row == col) ? 
					(matrix_i[row][col] - lambda) : matrix_i[row][col];
			}

			eigenvectors.resize(matrix_i.size());
			jordan_gaussian_transform(matrix_target, eigenvectors[index]);

			float eigsum_sq = 0;
			for (std::uint32_t v = 0; v < eigenvectors[index].size(); v++)
				eigsum_sq += std::pow(eigenvectors[index][v], 2.0);

			for (std::uint32_t v = 0; v < eigenvectors[index].size(); v++)
				eigenvectors[index][v] /= sqrt(eigsum_sq);

			eigenvalues[index] = std::sqrt(eigenvalues[index]);
		}

		return;
	}

	compute_evd(matrix_new, eigenvalues, eigenvectors, eig_count + 1);

	return;
}

void preprocessing::jordan_gaussian_transform(
	std::vector<std::vector<float>> matrix, std::vector<float>& eigenvector)
{
	const float eps = 0.000001; bool eigenv_found = false;
	for (std::uint32_t s = 0; s < matrix.size() - 1 && !eigenv_found; s++)
	{
		std::uint32_t col = s; float alpha = matrix[s][s];
		while (col < matrix[s].size() && alpha != 0 && alpha != 1)
			matrix[s][col++] /= alpha;

		for (std::uint32_t col = s; col < matrix[s].size() && !alpha; col++)
			std::swap(matrix[s][col], matrix[s + 1][col]);

		for (std::uint32_t row = 0; row < matrix.size(); row++)
		{
			float gamma = matrix[row][s];
			for (std::uint32_t col = s; col < matrix[row].size() && row != s; col++)
				matrix[row][col] = matrix[row][col] - matrix[s][col] * gamma;
		}

		std::uint32_t row = 0;
		while (row < matrix.size() &&
			(s == matrix.size() - 1 || std::fabs(matrix[s + 1][s + 1]) < eps))
			eigenvector.push_back(-matrix[row++][s + 1]);

		if (eigenvector.size() == matrix.size())
		{
			eigenv_found = true; eigenvector[s + 1] = 1;
			for (std::uint32_t index = s + 1; index < eigenvector.size(); index++)
				eigenvector[index] = (std::fabs(eigenvector[index]) >= eps) ? eigenvector[index] : 0;
		}
	}
}

void preprocessing::get_hermitian_matrix(std::vector<float> eigenvector,
	std::vector<std::vector<float>>& h_matrix)
{
	h_matrix.resize(eigenvector.size());
	for (std::uint32_t row = 0; row < eigenvector.size(); row++)
		h_matrix[row].resize(eigenvector.size());

	h_matrix[0][0] = 1 / eigenvector[0];
	for (std::uint32_t row = 1; row < eigenvector.size(); row++)
		h_matrix[row][0] = -eigenvector[row] / eigenvector[0];

	for (std::uint32_t row = 1; row < eigenvector.size(); row++)
		h_matrix[row][row] = 1;
}


void preprocessing::get_hermitian_matrix_inverse(std::vector<float> eigenvector,
	std::vector<std::vector<float>>& ih_matrix)
{
	ih_matrix.resize(eigenvector.size());
	for (std::uint32_t row = 0; row < eigenvector.size(); row++)
		ih_matrix[row].resize(eigenvector.size());

	ih_matrix[0][0] = eigenvector[0];
	for (std::uint32_t row = 1; row < eigenvector.size(); row++)
		ih_matrix[row][0] = -eigenvector[row];

	for (std::uint32_t row = 1; row < eigenvector.size(); row++)
		ih_matrix[row][row] = 1;
}

void preprocessing::get_inverse_diagonal_matrix(std::vector<std::vector<float>> matrix,
	std::vector<std::vector<float>>& inv_matrix)
{
	inv_matrix.resize(matrix.size());
	for (std::uint32_t index = 0; index < matrix.size(); index++)
	{
		inv_matrix[index].resize(matrix[index].size());
		inv_matrix[index][index] = 1.0 / matrix[index][index];
	}
}

void preprocessing::get_reduced_matrix(std::vector<std::vector<float>> matrix,
	std::vector<std::vector<float>>& r_matrix, std::size_t new_size)
{
	r_matrix.resize(new_size);
	std::size_t index_d = matrix.size() - new_size;
	std::uint32_t row = index_d, row_n = 0;
	while (row < matrix.size())
	{
		r_matrix[row_n].resize(new_size);
		std::uint32_t col = index_d, col_n = 0;
		while (col < matrix.size())
			r_matrix[row_n][col_n++] = matrix[row][col++];

		row++; row_n++;
	}
}

void preprocessing::matrix_by_matrix(std::vector<std::vector<float>> matrix1,
	std::vector<std::vector<float>>& matrix2, std::vector<std::vector<float>>& matrix3)
{
	matrix3.resize(matrix1.size());
	for (std::uint32_t row = 0; row < matrix1.size(); row++)
	{
		matrix3[row].resize(matrix1[row].size());
		for (std::uint32_t col = 0; col < matrix1[row].size(); col++)
		{
			matrix3[row][col] = 0.00;
			for (std::uint32_t k = 0; k < matrix1[row].size(); k++)	// TODO 
				matrix3[row][col] += matrix1[row][k] * matrix2[k][col];
		}
	}
}

void preprocessing::matrix_transpose(std::vector<std::vector<float>> matrix1, std::vector<std::vector<float>>& matrix2)
{
	matrix2.resize(matrix1.size());
	for (std::uint32_t row = 0; row < matrix1.size(); row++)
	{
		matrix2[row].resize(matrix1[row].size());
		for (std::uint32_t col = 0; col < matrix1[row].size(); col++) {
			cout << matrix1[col][row] << endl;
			matrix2[row][col] = matrix1[col][row];
		}
	}
}

pair<vector<vector<float>>, vector<float>> preprocessing::createDataMatrix(string* dirName)
{
	int img_num = NUM_EIGEN_FACES;
	for (int i = 0; i < img_num; i++)
	{
		this->images.push_back(imread(dirName[i], IMREAD_GRAYSCALE));
	}
	std::cout << "Creating data matrix from images ..." << std::endl;

	int img_h = images[0].rows;
	int img_w = images[0].cols;

	img_h_w = { img_h, img_w };
	vector<vector<float>> data(img_num, vector<float>(img_h * img_w * 3));

	vector<float> mean_data((img_h * img_w * 3));
	
	for (unsigned int i = 0; i < images.size(); i++)
	{
		for (int k = 0; k < images[i].rows; ++k) {
			for (int j = 0; j < images[i].cols; ++j) {
				data[i][( k * images[i].cols +j )] = static_cast<float>(images[i].at<uchar>(k, j));  // row, col
				mean_data[k * images[i].cols + j] += static_cast<float>(images[i].at<uchar>(k, j));
			}
		}
	}
	
	for (int j = 0; j < img_h * img_w * 3; j++) {
		mean_data[j] /= 8.0;
	}
	
	std::cout << " DONE" << std::endl;
	return { data, mean_data };
}

Mat preprocessing::PCA(pair<vector<vector<float>>, vector<float>> data_args) {
	// Calculate PCA of the data matrix
	cout << "Calculating PCA ...";

	vector<vector<float>> data = data_args.first;
	vector<float> mean_data = data_args.second;

	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[0].size(); j++) {
			data[i][j] -= mean_data[j];
		}
	}

	std::vector<std::vector<float>> u, v;
	std::vector<std::vector<float>> s;
	cout << data.size() << endl;
	svd(data, s, u, v);

	std::vector<std::vector<float>> transpose_u;
	matrix_transpose(u, transpose_u);
	std::vector<std::vector<float>> eigen_weight;
	matrix_by_matrix(transpose_u, data, eigen_weight);

	cout << " DONE" << endl;

	Mat output;
	resize(eigen_weight, output, Size(), this->img_h_w.first, this->img_h_w.second);

	return output;
}