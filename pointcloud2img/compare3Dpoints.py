import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from scipy import stats
import os
import pickle as pk
from utils import getRT
from loadModelDataset import loadDataset
from loadNVMformat import loadVSfMdata
from copy import deepcopy
import dense2image_reverse_ray as rev_map

MODELTYPE = "fixed_cam_params"
MODELTYPE2 = "refined_cam_params"
MODELTYPE3 = "pinhole_cam_params"
MODELTYPE4 = "simvsfm_cam_params"
MODELTYPE5 = "visualsfm_cam_params"
MODELTYPE6 = "exif_params"

select_model = MODELTYPE3

def getReprojError(ground_truth, reprj_points, reprj_points_type):
    
    totalErr =0

    for before, after in zip(ground_truth, reprj_points):
        # print("[",reprj_points_type, "]  gt ", before, "  reprj ", after)
        err = cv2.norm(before - after, cv2.NORM_L2)
        # print(reprj_points_type, " err : ", err)
        totalErr += np.sum(err*err)

    totalErr = np.sqrt(totalErr / len(reprj_points))
    print(reprj_points_type, " reprojection err : ", totalErr)

    return totalErr

if __name__ == "__main__":

    if MODELTYPE5 == select_model:
        dns_points, pairs_2d_3d, rt, cammatrix, rgb_data, distortion_params = loadVSfMdata(select_model)
    else:
        dns_points, pairs_2d_3d, rt, cammatrix, rgb_data, distortion_params = loadDataset(select_model)

    print("successfully model loaded")


    workdir = os.getcwd()
    model_dir = os.path.join(workdir, select_model)

    pts1 = pairs_2d_3d[:,1] # 3d point only

    res_dns_pnts = []

    avg_min_dist = 0
    

    dns_points = np.array(dns_points).astype(np.float32)
    
    tree = KDTree(dns_points)

    for pts in pairs_2d_3d:

        pnt_3d = np.array(pts[1]).astype(np.float32)
        dist, ind = tree.query([pnt_3d])
        dns_coor = dns_points[ind[0][0].astype(int)]
        avg_min_dist += dist[0]
        dns_pointss = np.concatenate((dist[0], dns_coor, pts)).tolist()
        res_dns_pnts.append(dns_pointss)
    
    hist_data = np.array([x[0] for x in res_dns_pnts])

    z_score = np.abs(stats.zscore(hist_data)) # z-score for hist data

    # Reference : https://www.ctspedia.org/do/view/CTSpedia/OutLier#:~:text=Any%20z%2Dscore%20greater%20than,standard%20deviations%20from%20the%20mean.
    # Any z-score greater than 3 or less than -3 is considered to be an outlier.
    outlier_idx = np.where(z_score > 1.96)
    dns_pnts_coor = np.array([x[1:4] for x in res_dns_pnts])
    outliers = np.array([dns_pnts_coor[x] for x in outlier_idx[0]]) # dns_point

    inlier_idx = np.where(z_score <= 1.96)
    inliers = np.array([res_dns_pnts[x] for x in inlier_idx[0]])
    inliers_distances = [hist_data[x] for x in inlier_idx[0]]
    inlier_2d = np.array([x[4] for x in inliers])
    inlier_3d = np.array([dns_pnts_coor[x] for x in inlier_idx[0]])
    inlier_3d_sparse = np.array([x[5] for x in inliers]) # sparse_point
    
    print('the number of outliers : ', len(outliers))
    print('the number of inliers : ', len(inliers))
    print("avg min dist (includes outliers)", avg_min_dist/ len(pts1))
    print("maximum dist (apart from outliers) : ", max(inliers_distances))
    q25, q75 = np.percentile(hist_data,[.25,.75])
    bin_width = 2*(q75 - q25)*len(hist_data)**(-1/3)
    bins = round((hist_data.max() - hist_data.min())/bin_width)

    # plt.hist(hist_data, bins = bins)
    # plt.boxplot(hist_data)
    # plt.show()
    
    #########
    ### check 2d points
    image = cv2.imread('IMG_0345.JPG', cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image2 = deepcopy(image)
    rvec = rt[:3,:3]
    tvec = rt[:,3]

    print("pairs_2d_3d ", len(pairs_2d_3d))
    for x in inlier_2d:
        mat_x = int(float(x[0]))
        mat_y = int(float(x[1]))
        
        cv2.circle(image, (mat_x, mat_y), 5, (255,0,0),1)
    
    ############
    ### IMG_0345 2번째 3d point에 매칭하는 점 실험 결과
    rep_2dpoints = []

    print("inlier_3d ", len(inlier_3d)) # dns
    reproj_inlier_3d, _ = cv2.projectPoints(inlier_3d, rvec, tvec, cammatrix, distortion_params)
    for coor in reproj_inlier_3d:
        cv2.circle(image2, (int(coor[0][0]), int(coor[0][1])), 5, (255,0,0),1)

    image3 = deepcopy(image2)

    print("outliers ", len(outliers))
    proj_outliers, _ = cv2.projectPoints(outliers, rvec, tvec, cammatrix, distortion_params)
    for coor in proj_outliers:
        cv2.circle(image2, (int(coor[0][0]), int(coor[0][1])), 5, (0,0,255),1)

    res = cv2.resize(image, (1040, 1040))
    res2 = cv2.resize(image2, (1040, 1040))
    # cv2.imshow("./result/original_2d.jpg", res)
    # cv2.imshow("./result/dns2image.jpg", res2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(model_dir,"result/original_2d.jpg"), res)
    cv2.imwrite(os.path.join(model_dir,"result/dns2image.jpg"), res2)
    
    ##### should move those lines upward in the future 
    with open(os.path.join(model_dir,"result/hist.txt"), "w") as f:
        for a in hist_data:
            a = a.astype(str)
            f.write(a+'\n')
    
    with open(os.path.join(model_dir,"result/inlier.pkl"), "wb") as g:
            pk.dump(inliers, g)

    ########solvePnP inliers
    #Canon EOS D6 Mark2 's distortion coefficients
    # 2 radial distortion
    # suppose NO tangential distortion
    
    _, rvec_refined, tvec_refined, _= cv2.solvePnPRansac(inlier_3d_sparse, inlier_2d, cammatrix, distortion_params,useExtrinsicGuess=True)

    inlier_refined_2d = []
    dist_with_refined_rt = []
    inlier_avg_min_dist = 0

    for inlier_pts in inlier_3d_sparse:

        inlier_pnt_3d = np.array(inlier_pts).astype(float)
        dist, ind = tree.query([inlier_pnt_3d])
        inlier_dns_coor = dns_points[ind[0][0].astype(int)]
        inlier_avg_min_dist += dist[0]
        
        # inlier_dns_coor = np.concatenate((inlier_dns_coor, [1.]))
        # inlier_dns_to_2d = np.matmul(inlier_rt, inlier_dns_coor)
        # inlier_dns_to_2d = np.matmul(cammatrix, inlier_dns_to_2d)

        # res = inlier_dns_to_2d[:] / inlier_dns_to_2d[2]
        # inlier_refined_2d.append(res[0:2])
        dist_with_refined_rt.append(dist[0])

    inliner_hist_data = np.array([x for x in dist_with_refined_rt])
    print("inlier_avg_min_dist : ", inlier_avg_min_dist / len(inlier_3d_sparse))

    with open(os.path.join(model_dir,"result/hist_refined.txt"), "w") as f:
        for a in inliner_hist_data:
            a = a.tolist()
            a = str(a[0])
            f.write(a+'\n')
    
    # plt.hist(inliner_hist_data, bins = bins)
    # plt.boxplot(inliner_hist_data)
    # plt.show()

    print("inlier_3d ", len(inlier_3d))
    
    reproj_inlier_3d_adj, _ = cv2.projectPoints(inlier_3d, rvec_refined, tvec_refined, cammatrix, distortion_params)
    for coor in reproj_inlier_3d_adj:
        cv2.circle(image3, (int(coor[0][0]), int(coor[0][1])), 5, (0,0,255),1)

    # Tinlier_refined_2d & reproj_inlier_3d distance comparison
    rprj_err = getReprojError(inlier_2d, reproj_inlier_3d, "adj BEFORE")
    adj_rprj_err = getReprojError(inlier_2d, reproj_inlier_3d_adj, "adj AFTER")
    print("Difference reprojection error (rprj_err - adj_rprj_err): ", rprj_err-adj_rprj_err)
    res3 = cv2.resize(image3, (1040, 1040))
    cv2.imshow("inlier_result" , res3)
    cv2.imwrite(os.path.join(model_dir,"result/inlier_result.jpg") , res3)
    cv2.waitKey()
    cv2.destroyAllWindows()

    refined_rt = np.zeros(shape=(3,3))
    cv2.Rodrigues(rvec_refined, refined_rt)
    refined_rt = cv2.hconcat((refined_rt,tvec_refined))
    print("rt : ", rt,"  refined_rt : ", refined_rt)
    args = (dns_points,pairs_2d_3d,refined_rt, cammatrix, rgb_data, max(inliers_distances), model_dir)

    rev_map.reverse_mapping(args)