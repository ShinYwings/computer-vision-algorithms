import numpy as np
import cv2
from sklearn.neighbors import KDTree
from utils import drange
from loadModelDataset import loadDataset
import pickle as pk
import os
import struct
import matplotlib.pyplot as plt

MODELTYPE = "fixed_cam_params"
MODELTYPE2 = "refined_cam_params"
MODELTYPE3 = "pinhole_cam_params"
MODELTYPE4 = "simvsfm_cam_params"
MODELTYPE5 = "visualsfm_cam_params"
MODELTYPE6 = "exif_params"

SELECTED_MODEL = MODELTYPE3

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def mapping(args):

    #Read Depth map
    cur_path= os.getcwd()
    depthmap_path = os.path.join(cur_path, "IMG_0345.JPG.geometric.bin")
    
    depth_map = read_array(depthmap_path)
    min_depth_percentile = 0
    max_depth_percentile =100
    min_depth, max_depth = np.percentile(
        depth_map, [min_depth_percentile, max_depth_percentile])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    depth_map = cv2.flip(depth_map, 1)
    depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    depth_map = cv2.resize(depth_map, (2080, 2080))
    # plt.figure()
    # plt.imshow(depth_map)
    # plt.title("depth map")
    # plt.show()
    # TODO res to depthmap
    res = np.zeros((2080, 2080, 3), dtype='uint8')
    # experiment_3dpoints = []
    # experiment_result= []
    
    kdtree = KDTree(np.array(tree_data))

    ## 각 픽셀에 해당하는 3d point 계산 
    # TODO implement inliers from compare3D points and check the circle creation
    for h in range(0,2080): # depthmap size
        print("current h is ", h)
        for w in range(0,2080):
            idx = [h, w, 1]
            idx = np.matmul(rt_inv, idx)
            idx = idx[:] / idx[3]
            
            ray_direction = idx[0:3] - cam_origin
            
            length = np.linalg.norm(ray_direction)
            ray_hat = ray_direction / length
            
            scale = depth_map[h][w]

            min = 99

            if scale > 3 and scale < 8:
                
                scale_range = scale-0.8
                while(scale_range < scale+0.8):

                    ray = scale_range * ray_hat + cam_origin
                    dist, ind = kdtree.query([ray])

                    if (dist < min and dist < 0.33700424361012217):  # max value yields from compare3Dpoints.py   # TODO modified dist condition
                        
                        res[h,w] = rgb_data[ind[0][0].astype(int)][::-1]
                        min = dist

                    scale_range+=0.005
                
        cv2.imshow("mapping_with_depthmap" , res)
        cv2.waitKey(1)
                
    return (res)

if __name__ == "__main__":
    
    # tree_data = dns_points , rgb_data = rgb value of the dns_points
    dns_points, pairs_2d_3d, rt, cammatrix, rgb_data, _ = loadDataset(SELECTED_MODEL) 
    
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
    
    args = (rt_inv, cam_origin, tree_data, rgb_data)

    res = mapping(args= args)
    
    res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
    res = cv2.flip(res, 1)
    res = cv2.resize(res, (1040,1040))

    # res2 = cv2.rotate(res2, cv2.ROTATE_90_CLOCKWISE)
    # res2 = cv2.flip(res2, 1)
    
    # with open("./result/no2_experiment_result.txt", "w") as f:
    #     for exp in experiment_result:
    #         f.write(exp)
    
    cv2.imshow("mapping_with_depthmap" , res)
    # cv2.imshow("reverse_noray2_experiment", res2)
    cv2.imwrite("mapping_with_depthmap.jpg", res)
    # cv2.imwrite("reverse_noray2_experiment.jpg", res2)
    cv2.waitKey()
    cv2.destroyAllWindows()