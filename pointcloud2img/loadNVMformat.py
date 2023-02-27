import numpy as np
import os
import sys
from utils import getRT
import pickle as pk

def loadVSfMdata(modeltype):
    
    workdir = os.getcwd()
    model_dir = os.path.join(workdir, modeltype)
    print('model_dir : ', model_dir)
    print("Load Dataset from the model")

    ##############################
    ### cameras.txt
    camMatrix = np.zeros((3,3), dtype=np.float32)
    print("Loading Intrinsic parameters...", end="  ")
    with open(os.path.join(model_dir,'cameras.txt'), 'r') as p:
        sen = p.read()
        intrinsic_params = sen.split(sep=' ')
        
    distortion_params = np.array(list(map(np.float32, [intrinsic_params[6], 0. ,0. ,0.])))

    camMatrix[0,0] = intrinsic_params[2]
    camMatrix[1,1] = intrinsic_params[4]
    camMatrix[0,2] = intrinsic_params[3]
    camMatrix[1,2] = intrinsic_params[5]
    camMatrix[2,2] = 1.

    print("Done.")
    
    ###################################
    ### points3D.txt of sparse set
    ###########################
    print("Pairing 2D points(image) to 3D points (sparse model)...")

    pairs_path = os.path.join(model_dir,'pairs_2d_3d.pkl')

    if os.path.isfile(pairs_path):
        with open(pairs_path, 'rb') as f:
            print("load pairs_data from ", pairs_path)
            pairs_2d_3d = pk.load(f)
    else:    
        print("Loading 3D points of sparse model...", end="  ")
        pairs_2d_3d = []

        image0345_index = 33
        with open(os.path.join(model_dir, 'points3D.txt'), 'r') as p:
            all = p.readlines()
            for idx, sen in enumerate(all):
                if idx == 0:
                    # sparse_points_size = int(sen)
                    continue
                else:
                    tmp = sen.split(sep=' ')
                    point3D_sparse = np.array(list(map(np.float32, tmp[0:3])))
                    # rgb = np.array(list(map(int, tmp[3:6])))
                    size_of_measurement = int(tmp[6])

                    for i in range(7,size_of_measurement*4+7, 4):
                        image_index = int(tmp[i])

                        if(image_index == image0345_index):
                            feature_index = int(tmp[i+1])
                            point2D = np.array(list(map(np.float32, tmp[i+2:i+4])))
                            
                            # print(image_index, " ", feature_index, " ", point2D)
                            pairs_2d_3d.append([point2D, point3D_sparse])
        pairs_2d_3d = np.array(pairs_2d_3d)
        with open(pairs_path, 'wb') as f:
            pk.dump(pairs_2d_3d, f)
    
    print("Done.")

    ###################################
    ### points3D.txt of dense set
    print("Loading 3D points of dense model...", end="  ")
    skip = 3
    dns_points = []
    dns_brg_color = []
    dns_a = []
    with open(os.path.join(model_dir,'points3D_dense.txt'), 'r') as p:
        dns_all = p.readlines()
        for sen in dns_all:
            if skip != 0:
                skip-=1
                continue
            else:
                dns_a.append(sen.split(sep='\n'))
    for s in dns_a:
        tmp = s[0].split(' ')
        dns_points.append(np.array(list(map(np.float32, tmp[1:4]))))  # TODO RGB 정보 뺌 뒤에 트리에서 필요없어서
        dns_brg_color.append(np.array(list(map(np.float32, tmp[4:7]))))
    print("Done.")
    
    #####################################################
    ### image.txt IMG_0345.JPG   (id 33)
    print("Loading image data...", end="  ")
    with open(os.path.join(model_dir,'image.txt'), 'r') as p:
        sen = p.read()
        img_metadata = sen.split(sep=' ')
    print(img_metadata)
    quaternion = [float(k) for k in img_metadata[1:5]]
    translation = [float(j) for j in img_metadata[5:8]]
    rt = getRT(quaternion, translation) 
    distortion_params[0] = img_metadata[8]
    # # TODO projectpoints experiment
    # quaternion = np.array(list(map(np.float32, img_metadata[1:5])))
    # translation = np.array(list(map(np.float32, img_metadata[5:8])))

    print("Done.")

    return dns_points, pairs_2d_3d, rt, camMatrix, dns_brg_color, distortion_params

# use for debugging
# if __name__=="__main__":
#     loadVSfMdata("visualsfm_cam_params")