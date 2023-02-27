#include "imageMatching.h"

void imageMatching::getTemplateImage(Mat input_image) {

	this->image_width = input_image.cols;
	this->image_height = input_image.rows;
	this->template_image = new uchar * [input_image.rows];

	for (int i = 0; i < this->image_height; i++) {
		this->template_image[i] = new uchar[input_image.cols];
	}

	for (int i = 0; i < input_image.rows; ++i) {
		for (int j = 0; j < input_image.cols; ++j) {
			this->template_image[i][j] = input_image.at<uchar>(i, j);  // row, col
		}
	}
}

pair<int, vector<PointInfo>> imageMatching::matchingStart(uchar*** pieces, pair<int, int> h_w) {

    int threadnum = this->threads_num;

    cout << "Start threading...!" << endl;
    vector<vector<PointInfo>> results(threadnum);
    vector<thread> workers;
    for (int i = 0; i < threadnum; i++) {
        workers.push_back(thread(&imageMatching::worker, this, pieces[i], h_w, &results[i]));
        cout << "worker " << i << " starts scanning!" << endl;
    }
    for (int i = 0; i < threadnum; i++) {
        workers[i].join();
        cout << "worker " << i << " has done scanning!" << endl;
    }
    cout << "Finish threading...!" << endl;
    
    // CCOEF_NORM 방법
    double maxCOEF{ -2.0 };
    int idx_of_max = 0; // about piece
    vector<PointInfo> ans_point;
    int idx{ 0 };
    for (auto iter = results.begin(); iter != results.end(); iter++) {

        vector<PointInfo> result = *iter;
        PointInfo highest_point = result.back();

        cout << " COEF is " << highest_point.getResult() << " at index " << idx << endl;
        if (maxCOEF < highest_point.getResult()) {
            idx_of_max = idx;
            ans_point = result;
            maxCOEF = highest_point.getResult();
        }
        idx++;
    }
    cout << "maxCOEF is " << maxCOEF << " at index is " << idx_of_max << endl;
    
    // SAD 방법
    /*
    int minSAD{ INF };
    int idx_of_max = 0; // about piece
    vector<PointInfo> ans_point;
    int idx{ 0 };
    for (auto iter = results.begin(); iter != results.end(); iter++) {

        vector<PointInfo> result = *iter;
        PointInfo highest_point = result.back();

        cout << " SAD is " << highest_point.getResult() << " at index " << idx << endl;
        if (minSAD > highest_point.getResult()){
            idx_of_max = idx;
            ans_point = result;
            minSAD = highest_point.getResult();
        }
        idx++;
    }
    cout <<  "minSAD is " << minSAD << " at index is " << idx_of_max << endl;
    */
    return { idx_of_max, ans_point };
}

void imageMatching::worker(uchar** data_original, pair<int, int> original_h_w, vector<PointInfo>* result) {

    //SumofAbsoluteDifferences(data_original, original_h_w, result);
    ccoefficient_normed(data_original, original_h_w, result);
}

void imageMatching::ccoefficient_normed(uchar** data_original, pair<int, int> original_h_w, vector<PointInfo>* result) {

    vector<PointInfo>& point = *result;
    int template_h = this->image_height;
    int template_w = this->image_width;
    int template_size = template_h * template_w;

    double best_position_coef = -2.0;
    int best_position_y{ 0 };
    int best_position_x{ 0 };

    bool calculateTemplateSummation = true; // 한번만 계산하면 됨
    
    int template_sum = 0;
    int original_sum = 0;
    double T_square_sum = 0.0;
    double I_square_sum = 0.0;
    double norm = 0.0;
    double z_temp = 0.0;
    double z_origin = 0.0;
    double coef = 0.0;

    // 타겟 이미지 안에서
    for (int original_y = 0; original_y < original_h_w.first - template_h; original_y++)
    {
        for (int original_x = 0; original_x < original_h_w.second - template_w; original_x++)
        {
            //템플릿 이미지와 타겟 이미지의 각각 summation
            for (int template_y = 0; template_y < template_h; template_y++)
            {
                for (int template_x = 0; template_x < template_w; template_x++)
                {
                    int original_pixel = static_cast<int>(data_original[(original_y + template_y)][(original_x + template_x)]);
                    int template_pixel = static_cast<int>(this->template_image[template_y][template_x]);

                    if (calculateTemplateSummation) {
                        template_sum += template_pixel;
                    }
                    original_sum += original_pixel;
                }
            }
            
            if (calculateTemplateSummation) {
                z_temp = static_cast<double>(template_sum) / static_cast<double>(template_size);
            }
            z_origin = static_cast<double>(original_sum) / static_cast<double>(template_size);
            
            //Similarity 계산
            for (int template_y = 0; template_y < template_h; template_y++)
            {
                for (int template_x = 0; template_x < template_w; template_x++)
                {
                    int template_pixel = static_cast<int>(this->template_image[template_y][template_x]);
                    int original_pixel = static_cast<int>(data_original[(original_y + template_y)][(original_x + template_x)]);

                    double T = static_cast<double>(template_pixel) - z_temp;
                    double I = static_cast<double>(original_pixel) - z_origin;

                    T_square_sum += pow(T, 2.0);
                    I_square_sum += pow(I, 2.0);
                    coef += (T * I); // 이게 수식...
                    //coef += sqrt(T * I); // 왜 이게 잘나오는거야 ...?????????  근데  T 가 음수 (불일치) 일때 SQRT(NEG) 라서 ABORT함
                }
            }
            norm = sqrt(T_square_sum * I_square_sum);

            coef /= norm; // 정규화된 상관계수

            // 최대 coef 지점 찾기  
            if (best_position_coef < coef)
            {   
                best_position_coef = coef;
                best_position_y = original_y;
                best_position_x = original_x;
            }

            calculateTemplateSummation = false;

            //reset
            original_sum = 0;
            T_square_sum = 0.0;
            I_square_sum = 0.0;
            coef = 0.0;
        }
    }
    //여기가 result 꺼낼때 맨 처음에 pop_back()해야하는 이유, 여기는 vector가 필요없는데 다른 방법들이랑 맞춰주기 위해서 사용
    PointInfo pointInfo(best_position_x, best_position_y, best_position_coef);
    point.push_back(pointInfo);
}

void imageMatching::SumofAbsoluteDifferences(uchar** data_original, pair<int, int> original_h_w, vector<PointInfo>* result) {

    vector<PointInfo>& point = *result;
    int template_h = this->image_height;
    int template_w = this->image_width;

    int best_position_sad = INF;
    int best_position_y{ 0 };
    int best_position_x{ 0 };

    for (int original_y = 0; original_y < original_h_w.first - template_h; original_y++)
    {
        for (int original_x = 0; original_x < original_h_w.second - template_w; original_x++)
        {
            int SAD = 0;

            //템플릿 이미지 스캔
            for (int template_y = 0; template_y < template_h; template_y++)
            {
                for (int template_x = 0; template_x < template_w; template_x++)
                {
                    int original_pixel = static_cast<int>(data_original[(original_y + template_y)][(original_x + template_x)]);  //공식 
                    int template_pixel = static_cast<int>(this->template_image[template_y][template_x]);

                    SAD += abs(original_pixel - template_pixel);
                }
            }

            // 최소 SAD 지점 찾기  
            if (best_position_sad > SAD)
            {
                best_position_sad = SAD;
                best_position_y = original_y;
                best_position_x = original_x;
            }

            PointInfo pointInfo(original_x, original_y, SAD);
            point.push_back(pointInfo);
        }
    }

    //여기가 result 꺼낼때 맨 처음에 pop_back()해야하는 이유, 최고 점을 마지막 벡터에다가 저장함
    PointInfo pointInfo(best_position_x, best_position_y, best_position_sad);
    point.push_back(pointInfo);
}