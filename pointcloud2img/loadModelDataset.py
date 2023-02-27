import numpy as np
import os
import sys
from utils import getRT
import pickle as pk

def loadDataset(modeltype):
    
    workdir = os.getcwd()
    model_dir = os.path.join(workdir, modeltype)
    print('model_dir : ', model_dir)
    print("Load Dataset from the model")

    ##############################
    ### cameras.txt
    camMatrix = np.zeros((3,3), dtype=np.float32)
    print("Loading Intrinsic parameters...", end="  ")
    with open(os.path.join(model_dir,'cameras.txt'), 'r') as p:
        all = p.readlines()
        for idx, sen in enumerate(all):
            if idx == 3:
                intrinsic_params = sen.split(sep=' ')
                if len(intrinsic_params) is 12:
                    print("OPENCV MODEL, current {num}".format(num =len(intrinsic_params))) 
                    intrinsic_params[11] = intrinsic_params[11].replace('\n', '')
                    distortion_params = np.array(list(map(np.float32, intrinsic_params[8:12])))
                elif modeltype == "simvsfm_cam_params":
                    print("SIMPLE_RADIAL MODEL, current {num}".format(num =len(intrinsic_params))) 
                    distortion_params = np.array([intrinsic_params[7], 0., 0., 0.], dtype=np.float32)
                else:
                    print("Not enough the number of camera parameters, current {num}".format(num =len(intrinsic_params))) 
                    # exit(0)
                    distortion_params = np.array([0., 0., 0., 0.], dtype=np.float32)

    camMatrix[0,0] = intrinsic_params[4]
    camMatrix[1,1] = intrinsic_params[5]
    camMatrix[0,2] = intrinsic_params[6]
    camMatrix[1,2] = intrinsic_params[7]
    camMatrix[2,2] = 1.

    print("Done.")
    
    ###################################
    ### points3D.txt of sparse set
    print("Loading 3D points of sparse model...", end="  ")
    skip = 3
    points = []
    a = []
    with open(os.path.join(model_dir, 'points3D.txt'), 'r') as p:
        all = p.readlines()
        for sen in all:
            if skip != 0:
                skip-=1
                continue
            else:
                a.append(sen.split(sep='\n'))

    for s in a:
        points.append(np.array(list(map(np.float32, s[0].split(' ')))))
    
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
    ### image.txt IMG_0345.JPG   (id 8)
    print("Loading image data...", end="  ")
    with open(os.path.join(model_dir,'image.txt'), 'r') as p:
        all = p.readlines()
        for idx, sen in enumerate(all):
            if idx == 0:
                img_metadata = sen.split(sep=' ')
            else:
                point_2d = sen.split(sep=' ')
    point_2d = np.reshape(point_2d, ((int)(np.size(point_2d)/3), 3))
    
    cor = []
    for i in point_2d.tolist():
        if(i[2] != '-1'):
            cor.append(np.array(list(map(np.float32, i))))

    quaternion = [float(k) for k in img_metadata[1:5]]
    translation = [float(j) for j in img_metadata[5:8]]
    rt = getRT(quaternion, translation) 
    # # TODO projectpoints experiment
    # quaternion = np.array(list(map(np.float32, img_metadata[1:5])))
    # translation = np.array(list(map(np.float32, img_metadata[5:8])))
    
    print("Done.")

    ###########################
    print("Pairing 2D points(image) to 3D points (sparse model)...", end="  ")

    pairs_path = os.path.join(model_dir,'pairs_2d_3d.pkl')

    if os.path.isfile(pairs_path):
        with open(pairs_path, 'rb') as f:
            print("load pairs_data from ", pairs_path)
            pairs_2d_3d = pk.load(f)
        print("Done.")
    else:    
        pairs_2d_3d = []
        for x in points:
            for y in cor:
                if x[0] == y[2]:
                    pairs_2d_3d.append([y[0:2], x[1:4]])
        pairs_2d_3d = np.array(pairs_2d_3d)

        
        with open(pairs_path, 'wb') as f:
            pk.dump(pairs_2d_3d, f)
        print("Done.")
    
    return dns_points, pairs_2d_3d, rt, camMatrix, dns_brg_color, distortion_params