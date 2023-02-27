#include <opencv2/opencv.hpp>
#include <iostream>
#include "scanning.h"
#include "imageMatching.h"
#include "preprocessing.h"
#include "resource.h"

using namespace cv;
using namespace std;


const int THREADS_NUM = 10;
const int BOLD = 5;
const string DIRNAME = "images\\";

string* templatepath = new string[10]{ ".\\asset\\template\\suspector1.jpg","asset\\template\\suspector2.jpg", 
"asset\\template\\suspector3.jpg",".\\asset\\template\\suspector4.jpg",
"asset\\template\\suspector5.jpg",".\\asset\\template\\suspector6.jpg",
                    "asset\\template\\suspector7.jpg","asset\\template\\suspector8.jpg" };

int main()
{
    Mat img_color = imread("asset\\image\\target1.jpg", IMREAD_COLOR);
    Mat img_original = imread("asset\\image\\target1.jpg", IMREAD_GRAYSCALE);
    Mat img_template = imread("asset\\template\\suspector1.jpg", IMREAD_GRAYSCALE);

    // TODO target image / thread < template image 이면 못하게 예외처리

    // PCA로 Wally 의 모습들을 eigenface화 한것
    //preprocessing eigenface;
    //pair<vector<vector<float>>, vector<float>> data = eigenface.createDataMatrix(templatepath);
    //Mat img_template = eigenface.PCA(data);

    imageMatching Matcher(THREADS_NUM);
    Matcher.getTemplateImage(img_template);
    int THRESHOLD = Matcher.getImageHeight() / 2 +1;
    scanning Scanner(THRESHOLD, THREADS_NUM); // 이미지간 얼만큼 겹치게 할지 정함 (ex) 50이면 100pixel 씩 겹침 (4개로 나눠서 작업해야하기 때문에 템플릿이 짤리는 부분 있는 것도 고려)
    
    Scanner.getImage(img_original);
    Scanner.divideImage();
    uchar*** pieces = Scanner.imageOut();

    pair<int, int> original_h_w = {img_original.rows, img_original.cols};
    pair<int, int> template_h_w = { img_template.rows, img_template.cols};
    pair<int, int> piece_h_w = { Scanner.debug().first, original_h_w.second};
    
    int* sp = Scanner.getStartingPoint_each_piece();
    pair<int, vector<PointInfo>> ans = Matcher.matchingStart(pieces, piece_h_w);
    
    int idx = ans.first;
    vector<PointInfo> point = ans.second;
    int best_position_x = point.back().getX();
    int best_position_y = point.back().getY();
    double best_position_result = point.back().getResult();
    
    point.pop_back(); // 최고점 제거

    //COEF_NORM 방법
    // Drawing Rectangle 함수로 만들기 귀찬아서 다 때려박음
    for (int x = best_position_x- BOLD; x < best_position_x + template_h_w.second+ BOLD; x++){
        for (int y = best_position_y + sp[idx]- BOLD; y < best_position_y + sp[idx] + template_h_w.first+ BOLD; y++){

            if ((x >= best_position_x && x < best_position_x + template_h_w.second) && (y >= best_position_y + sp[idx] && y < best_position_y + sp[idx] + template_h_w.first)) {
                continue;
            }else{
                img_color.at<Vec3b>(y, x)[0] = 255;
                img_color.at<Vec3b>(y, x)[1] = 0;
                img_color.at<Vec3b>(y, x)[2] = 0;
            }
        }
    }
    
    // SAD 방법
    /*
    for (int i = 0; i < point.size(); i++)
    {
        if (abs(best_position_result - point[i].getResult()) < 100) {

            for (int x = point[i].getX()- BOLD; x < point[i].getX() + template_h_w.second+ BOLD; x++){
                for (int y = point[i].getY() + sp[idx]- BOLD; y < point[i].getY() + sp[idx] + template_h_w.first + BOLD; y++){

                    if ((x >= best_position_x && x < best_position_x + template_h_w.second) && (y >= best_position_y + sp[idx] && y < best_position_y + sp[idx] + template_h_w.first)) {
                        continue;
                    }else{
                        img_color.at<Vec3b>(y, x)[0] = 255;
                        img_color.at<Vec3b>(y, x)[1] = 0;
                        img_color.at<Vec3b>(y, x)[2] = 0;
                    }
                 }
             }
         }
    }
    */

    imshow("result", img_color);
    waitKey(0);
    destroyAllWindows();

    return 0;
}