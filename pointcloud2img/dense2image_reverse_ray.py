import numpy as np
import cv2
from sklearn.neighbors import KDTree
from utils import drange
from loadModelDataset import loadDataset
import pickle as pk
import os

MODELTYPE = "fixed_cam_params"
MODELTYPE2 = "refined_cam_params"
MODELTYPE3 = "pinhole_cam_params"
MODELTYPE4 = "simvsfm_cam_params"
MODELTYPE5 = "visualsfm_cam_params"
MODELTYPE6 = "exif_params"

SELECTED_MODEL = MODELTYPE3

def mapping(args):

    rt_inv, cam_origin, tree_data, rgb_data, max_dist, model_dir = args

    # TODO make (500, 500) image and check the range of the passing rays
    # ( experiment 1 ) check the existence of 3d points in the image at bottom left side
    res = np.zeros((2080, 2080, 3), dtype='uint8')
    res2 = np.zeros((600, 600, 3), dtype='uint8')
    experiment_3dpoints = []
    experiment_result= []
    
    kdtree = KDTree(np.array(tree_data))

    ## 각 픽셀에 해당하는 3d point 계산 
    # TODO implement inliers from compare3D points and check the circle creation
    for h in range(0,2080):
        print("current h is ", h)
        for w in range(0,2080):
            idx = [h, w, 1]
            idx = np.matmul(rt_inv, idx)
            idx = idx[:] / idx[3]
            
            ray_direction = idx[0:3] - cam_origin
            
            length = np.linalg.norm(ray_direction)
            ray_hat = ray_direction / length
            # print("ray_direction : ", ray_direction)
            
            # Adjustable parameter: ray scale,  intercept (first point or origin),  threshold (outlier)  
            min = 99
            for scale in drange(-8,8,'0.1'): # z value doesnt exceed 10 within 3d points of building  TODO (before -8, 8)
                
                # scale_refine = scale * np.array([1.5 , 1., 1.5]) # TODO scale adjustment
                ray = scale * ray_hat + idx[0:3]
                dist, ind = kdtree.query([ray])

                if (dist < min and dist < max_dist):  # max value yields from compare3Dpoints.py 
                    
                    min = dist
                    res[h,w] = rgb_data[ind[0][0].astype(int)][::-1]

                    if w < 600 and w >= 1480:
                        res_w = w - 1480 # TODO make variable
                        res2[h,res_w] = rgb_data[ind[0][0].astype(int)][::-1]
                        experiment_3dpoints.append(tree_data[ind[0][0]])
                        tmp = "experiment[{h},{w}] : {d} and color {c}".format(h=h, w=res_w, d= tree_data[ind[0][0]], c=res2[h,res_w])
                        experiment_result.append(tmp)
                        print(tmp)

    return (res, res2, experiment_result)

def reverse_mapping(args):
    
    dns_points, pairs_2d_3d, rt, cammatrix, rgb_data, max_dist, model_dir = args
    # tree_data = dns_points , rgb_data = rgb value of the dns_points
    # dns_points, pairs_2d_3d, rt, cammatrix, rgb_data, _ = loadDataset(SELECTED_MODEL) 
    
    # cur_path = os.getcwd()
    # model_path = os.path.join(cur_path, "./{modeltype}".format(modeltype=SELECTED_MODEL))
    # inlier_path = os.path.join(model_path, "result/inlier.pkl")
    
    # TODO quotes for whole dataset test
    tree_data = dns_points
    # if os.path.isfile(inlier_path):
    #     with open(inlier_path, "rb") as f:
    #         tree_data=pk.load(f)
    #         tree_data = np.array([x[1:4] for x in tree_data])
    #     print("successfully loaded 3D inlier dense points")
    # else:
    #     print("failed to load inliers, get whole 3D dense points")
    #     tree_data = dns_points

    # distortion 고려해줘야하나?
    image = cv2.imread('IMG_0345.JPG', cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
   
    inv_rot = np.linalg.inv(rt[0:3, 0:3])
    translation = rt[:,3]
    cam_origin = -(np.matmul(inv_rot, translation))
    print("cam_origin : ", cam_origin)
    matt = np.matmul(cammatrix, rt)
    rt_inv = np.linalg.pinv(matt)
    
    args = (rt_inv, cam_origin, tree_data, rgb_data, max_dist)

    res, res2, experiment_result = mapping(args= args)
    
    res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
    res = cv2.flip(res, 1)
    res = cv2.resize(res, (1040,1040))

    res2 = cv2.rotate(res2, cv2.ROTATE_90_CLOCKWISE)
    res2 = cv2.flip(res2, 1)
    
    with open(os.path.join(model_dir,"result/experiment_result.txt"), "w") as f:
        for exp in experiment_result:
            f.write(exp)
    
    cv2.imshow("reverse_ray" , res)
    cv2.imshow("reverse_rayexperiment", res2)
    cv2.imwrite(os.path.join(model_dir,"reverse_ray.jpg"), res)
    cv2.imwrite(os.path.join(model_dir,"reverse_ray_experiment.jpg"), res2)
    cv2.waitKey()
    cv2.destroyAllWindows()