import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
import os
import pickle as pk
from utils import getRT
from loadModelDataset import loadDataset

if __name__=="__main__":

    img1 = cv2.imread('-1_1_0.02_0.207.jpg', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('IMG_0345.JPG', cv2.IMREAD_COLOR)
    img2 = cv2.resize(img2, (1040,1040))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp, dsc = sift.detectAndCompute(gray,None)
    kp2, dsc2 = sift.detectAndCompute(gray2,None)

    print(len(dsc2))
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(dsc, dsc2, 2)

    ratio_thresh = 0.6
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance: # 1st short dist < 2nd short dist * 0.65    = less ratio thresh, more confident matching 
            good_matches.append(m)

    # TODO descriptor?   3d points in 2d_3d_pairs?  !!decision boundary!!

    #-- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, kp, img2, kp2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("matches",img_matches)
    cv2.waitKey()
    # img2=cv2.drawKeypoints(gray2,kp, img2)
    
    # print(len(kp))

    cv2.imwrite("matches.jpg", img_matches)

    print (len(good_matches))

    res3 = np.zeros((1040,1040))
    for i in range(0,1040):
        for j in range(0,1040):
            res3[i][j] = gray[i][j] - gray2[i][j]
            
    
    plt.figure()
    plt.imshow(res3)
    plt.title("diff")
    plt.show()
