```cpp
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

    cout << U.size() << S.size() << VT.size() << endl;
    
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
```