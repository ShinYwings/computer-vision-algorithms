import numpy as np
import cv2
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from loadModelDataset import loadDataset
from loadNVMformat import loadVSfMdata

MODELTYPE = "fixed_cam_params"
MODELTYPE2 = "refined_cam_params"
MODELTYPE3 = "pinhole_cam_params"
MODELTYPE4 = "simvsfm_cam_params"
MODELTYPE5 = "visualsfm_cam_params"

if __name__ == "__main__":
    
    select_model = MODELTYPE3

    if MODELTYPE5 == select_model:
        dns_points, pairs_2d_3d, rt, cammatrix, dns_brg_color, distortion_params = loadVSfMdata(select_model)
    else:
        dns_points, pairs_2d_3d, rt, cammatrix, dns_brg_color, distortion_params = loadDataset(select_model)
    
    print("successfully model loaded")

    ###########################
    src_image = cv2.imread('IMG_0345.JPG', cv2.IMREAD_GRAYSCALE)
    wh = src_image.shape
    image = cv2.cvtColor(src_image, cv2.COLOR_GRAY2BGR)
    
    res = np.ones((wh[0], wh[1], 3), dtype="uint8")
    
    ## 각 픽셀에 해당하는 3d point 계산 
    rvec = rt[:3,:3]
    tvec = rt[:,3]
    dns_points = np.array(dns_points)
    proj, _ = cv2.projectPoints(dns_points, rvec, tvec, cammatrix, distortion_params)
    for idx, coor in enumerate(proj):

        if (coor[0][0] >= 0 ) & (coor[0][1] >= 0) & (coor[0][0] < 2080) & (coor[0][1] < 2080):
            res[int(coor[0][0]), int(coor[0][1])] = dns_brg_color[idx][::-1]
    
    # for pnt in dns_points:

    #     pos = np.concatenate((pnt[0], [1]))
        
    #     # print('pos:', pos)
    #     campos = np.matmul(rt,pos)
    #     uv = np.matmul(cammatrix, campos)
    #     norm_uv = uv[:]/ uv[2]
    #     # print('uv: ', norm_uv)
    #     # matt = np.matmul(cammatrix, rt)
    #     # rt_inv = np.linalg.pinv(matt)
    #     # xyz_mod = np.matmul(rt_inv, norm_uv)
    #     # xyz_norm = xyz_mod[:] / xyz_mod[3]
    #     # print('xyz: ', xyz_norm)
    #     # test = np.matmul(rt, xyz_norm)
    #     # test = np.matmul(cammatrix, test)
    #     # test = test[:]/test[2]
    #     # print('test ', test)

        

    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    # cv2.circle(image, (1000, 436), 1, (0,0,255), 5)
    image = cv2.resize(image, (1040,1040))
    
    res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
    res = cv2.flip(res, 1)
    # cv2.circle(res, (1000, 436), 1, (0,0,255), 5)
    # res = cv2.resize(res, (1040,1040))
    
    cv2.imshow("dense2image" , res)
    cv2.imwrite("dense2image.jpg", res)
    cv2.waitKey()
    cv2.destroyAllWindows()