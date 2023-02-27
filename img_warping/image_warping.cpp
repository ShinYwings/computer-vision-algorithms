#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <cmath>

using namespace std;
using namespace cv;

const int MAX = 99999;

Mat img1;
Mat img2;
vector<Point2f> img1_pnts;
vector<Point2f> img2_pnts;
bool isStartSet = 0;

static void onMouse (int event, int x, int y, int, void* ptr)
{
    if((img1_pnts.size() == 4) && (img2_pnts.size() == 4) && (isStartSet == 0))
    {
        isStartSet = 1;
        printf ("START\n");
        return;
    }

    Mat *pMat = (Mat*)ptr;
    Mat img = Mat(*pMat);

    if (event == EVENT_LBUTTONDOWN)
    {
        vector<Point2f>& pnts = ((&img1 == ptr)? img1_pnts: img2_pnts);
        if (pnts.size() < 4)
        {
            Point2f startPnt(x,y);
            circle(img, startPnt, 7, cv::Scalar (255, 0, 0));
            pnts.push_back(startPnt);
        } 
    }
}

// Get homography with SVD
Mat getHomography(const vector<Point2f>& img1_pnt, const vector<Point2f>& img2_pnt)
{
    int x1 = img1_pnt[0].x;
    int y1 = img1_pnt[0].y;
    int x2 = img1_pnt[1].x;
    int y2 = img1_pnt[1].y;
    int x3 = img1_pnt[2].x;
    int y3 = img1_pnt[2].y;
    int x4 = img1_pnt[3].x;
    int y4 = img1_pnt[3].y;
    int xp1 = img2_pnt[0].x; 
    int yp1 = img2_pnt[0].y;
    int xp2 = img2_pnt[1].x;
    int yp2 = img2_pnt[1].y;
    int xp3 = img2_pnt[2].x;
    int yp3 = img2_pnt[2].y;
    int xp4 = img2_pnt[3].x;
    int yp4 = img2_pnt[3].y;

    Mat H = (Mat_<float>(8,9) << 
        -x1,  -y1,  -1,   0,    0,    0,   x1*xp1,   y1*xp1,   xp1,
        0,    0,    0, -x1,   -y1,  -1,   x1*yp1,   y1*yp1,   yp1,
        -x2,  -y2,  -1,   0,    0,    0,   x2*xp2,   y2*xp2,   xp2,
        0,    0,   0,  -x2,   -y2,  -1,   x2*yp2,   y2*yp2,   yp2,
        -x3,  -y3,  -1,   0,    0,    0,   x3*xp3,   y3*xp3,   xp3,
        0,    0,    0, -x3,   -y3,  -1,   x3*yp3,   y3*yp3,   yp3,
        -x4,  -y4,   -1,  0,    0,    0,   x4*xp4,   y4*xp4,   xp4,
        0,    0,    0,  -x4,  -y4,  -1,   x4*yp4,   y4*yp4,   yp4);    
    
    Mat U,S,VT;
    SVDecomp(H,U,S,VT, SVD::FULL_UV); // H = U S V^T   [1x8] [8x8] [9x9]
    transpose(VT,VT);

    Mat col_vec = VT.col(8);
    col_vec = col_vec / col_vec.at<float>(8,0);
    Mat transform(Size(3,3), CV_32FC1);
    
    transform.at<float>(0,0) = col_vec.at<float>(0,0);
    transform.at<float>(0,1) = col_vec.at<float>(1,0);
    transform.at<float>(0,2) = col_vec.at<float>(2,0);
    transform.at<float>(1,0) = col_vec.at<float>(3,0);
    transform.at<float>(1,1) = col_vec.at<float>(4,0);
    transform.at<float>(1,2) = col_vec.at<float>(5,0);
    transform.at<float>(2,0) = col_vec.at<float>(6,0);
    transform.at<float>(2,1) = col_vec.at<float>(7,0);
    transform.at<float>(2,2) = col_vec.at<float>(8,0);

    return transform;
}

void warp(const Mat& src1, Mat& src2, const Mat& H)
{
    Mat H_inv= H.inv();

    for(int h=0; h< src2.rows; h++){
        for(int w=0; w <src2.cols; w++){
            
            Point3f coor(w,h,1);
            Mat tmp_coor = (H_inv * Mat(coor));
            tmp_coor /= tmp_coor.at<float>(2,0);

            Point trans_coor((cvRound(tmp_coor.at<float>(0,0))), (cvRound(tmp_coor.at<float>(1,0))));

            float tx = (cvRound(trans_coor.x));
            float ty = (cvRound(trans_coor.y));
            float a = trans_coor.x - tx;
            float b = trans_coor.y - ty;
            
            if(tx >= 0 && tx < src2.cols && ty >= 0 && ty < src2.rows)
            {   
                // Bilinear Interpolation (BGR)
                for(int i=0;i<3;i++)
                {
                    src2.at<Vec3b>(h, w)[i] = cvRound((((1.0 - a) * (1.0 - b)) * src1.at<Vec3b>(ty, tx)[i])
                        + ((a * (1.0 - b)) * src1.at<Vec3b>(ty, tx+1)[i])
                        + ((a * b) * src1.at<Vec3b>(ty+1, tx+1)[i])
                        + (((1.0 - a) * b) * src1.at<Vec3b>(ty+1, tx))[i]);
                }
            }
        }
    }
}

int main()
{
    img1 = imread("src.jpg", IMREAD_COLOR);
    img2 = imread("dst.jpg", IMREAD_COLOR);

    if (img1.empty() || img2.empty())
		return -1;

	resize(img2, img2, Size(img2.cols /8, img2.rows /8));

    Point2f img1_scale((float)img2.cols/img1.cols , (float)img2.rows/img1.rows);
    
    cout << img1_scale << endl;
    Mat rescaled_img1; 
	resize(img1, rescaled_img1, Size(img2.cols, img2.rows));

    Mat src1, src2;
    img1.copyTo(src1);
    img2.copyTo(src2);

    while (!isStartSet)
    {
        imshow ("src", img1);
        imshow ("dst", img2);
        setMouseCallback ("src", onMouse, (void*)&img1);
        setMouseCallback ("dst", onMouse, (void*)&img2);
        waitKey(30);
    }

    for(int i=0; i<4; i++){
        img1_pnts[i].x *= img1_scale.x;
        img1_pnts[i].y *= img1_scale.y;
    }

    // Mat H = getPerspectiveTransform(img1_pnts, img2_pnts);
    Mat H = getHomography(img1_pnts, img2_pnts);

    cout << "[Homograhy]\n" << H <<  endl;
    
    // Reverse Mapping + Bilinear Interpolation
    // warpPerspective(src1, src2, H, src2.size(), INTER_LINEAR + WARP_INVERSE_MAP);
    warp(rescaled_img1, src2, H);

    imshow("res", src2);
    waitKey();
    destroyAllWindows();

    return 0;
}