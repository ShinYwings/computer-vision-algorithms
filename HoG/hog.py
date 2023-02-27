#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:45:20 2021

@author: ubuntu
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

def octagon():
    return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#img = octagon() #cv2.imread('/home/somin/Desktop/ui4_b.png', cv2.IMREAD_GRAYSCALE) #octagon()

from scipy.ndimage.interpolation import zoom
# make the image bigger to compute the histogram
#img = zoom(octagon(), 20)

def hog(img):
    # srcimg = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)  #apple.jpg Lenna.png
    srcimg = img
    #img = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5)

    img = np.array(srcimg, dtype=np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, ori = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    

    # plt.figure()
    # plt.title('gradients and magnitude')

    # plt.subplot(141)
    # plt.imshow(img, cmap=plt.cm.get_cmap('gray'))

    # plt.subplot(142)
    # plt.imshow(gx, cmap=plt.cm.get_cmap('gray'))

    # plt.subplot(143)
    # plt.imshow(gy, cmap=plt.cm.get_cmap('gray'))

    # plt.subplot(144)
    # plt.imshow(mag, cmap=plt.cm.get_cmap('gray'))


    # Show the orientation deducted from gradient
    # plt.figure()
    # plt.title('orientations')
    # plt.imshow(ori)
    # plt.pcolor(ori)
    # plt.colorbar()
    # plt.show()
    
    cell_size  = 8  # pixel (2 pixel * 2 pixel)

    bx = 2  # cell (4 pixel * 4 pixel)
    by = 2

    # num of cells in x, y axis
    sy = img.shape[0] // cell_size
    sx = img.shape[1] // cell_size
    
    nbins = 9
    signed_orientation = False # 0: 180, 1: 360

    if signed_orientation:
        max_angle = 360
    else:
        max_angle = 180

    b_step = max_angle/nbins

    # orientation first index
    b0 = (ori % max_angle) // b_step
    b0[np.where(b0>=nbins)]=0

    # orientation second index
    b1 = b0 + 1
    b1[np.where(b1>=nbins)]=0

    # parts
    b = np.abs(ori % b_step) / b_step

    # pixel histogram
    temp_coefs = np.zeros((img.shape[0], img.shape[1], nbins))
    for i in range(nbins):
        temp_coefs[:, :, i] += np.where(b0==i, (1 - b), 0)
        temp_coefs[:, :, i] += np.where(b1==i, b, 0)

    pixel_histogram = np.multiply(temp_coefs, np.expand_dims(mag, axis=2))
    # print('pixel_histogram: ', pixel_histogram.shape)
    
    # cell histogram
    histogram = np.zeros((sy, sx, 9))
    for x in range(sx):
        for y in range(sy):
            cell = pixel_histogram[cell_size*y:cell_size*y+cell_size, cell_size*x:cell_size*x+cell_size, :].sum(axis=(0,1))
            histogram[y, x, :] += cell
    # print('histogram: ', histogram.shape, histogram.sum())

    # normalize
    eps = 1e-5

    n_blocksx = (sx - bx) + 1
    n_blocksy = (sy - by) + 1
   
    # normalised_blocks = np.zeros((n_blocksy, n_blocksx, nbins))
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, bx, by, nbins))
    
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            
            # if (x+1) == n_blocksx and (y+1) == n_blocksy: # x,y가 끝일때
            #     normalised_blocks[y, x, :] = np.clip(block[-1, -1, :] / np.sqrt(np.sum(block**2) + eps**2), 0, 0.2)

            # elif (x+1) == n_blocksx:
            #     normalised_blocks[y, x, :] = np.clip(block[0, -1, :] / np.sqrt(np.sum(block**2) + eps**2), 0, 0.2)

            # elif (y+1) == n_blocksy:
            #     normalised_blocks[y, x, :] = np.clip(block[-1, 0, :] / np.sqrt(np.sum(block**2) + eps**2), 0, 0.2)
                
            # else:
                # print("y+by =", y+by)
            block = histogram[y:y + by, x:x + bx, :]
            # normalised_blocks[y, x, :] = np.clip(block[0, 0, :] / np.sqrt(np.sum(block**2) + eps**2), 0, 0.2)
            normalised_blocks[y, x, :] = np.minimum(block / np.sqrt(np.sum(block**2) + eps**2), 0.2)
            
            normalised_blocks[y, x, :] /= np.sqrt(np.sum(normalised_blocks[y, x, :]**2) + eps**2)

    def visualise_histogram(hist, csx, csy, signed_orientation=False):

        from skimage import draw
        
        if signed_orientation:
            max_angle = 2*np.pi
        else:
            max_angle = np.pi

        n_cells_y, n_cells_x, nbins = hist.shape
        sx, sy = n_cells_x*csx, n_cells_y*csy
        center = csx//2, csy//2
        b_step = max_angle / nbins

        radius = min(csx, csy) // 2 - 1
        hog_image = np.zeros((sy, sx), dtype=float)

        # print('hog image: ', hog_image.shape)
        for x in range(n_cells_x):
            for y in range(n_cells_y):
                for o in range(nbins):
                    centre = tuple([y * csy + csy // 2, x * csx + csx // 2])
                    dx = radius * np.cos(o*nbins)
                    dy = radius * np.sin(o*nbins) #((o*b_step)*np.pi/180) #
                    rr, cc = draw.line(int(centre[0] - dy),
                                    int(centre[1] - dx),
                                    int(centre[0] + dy),
                                    int(centre[1] + dx))

                    hog_image[rr, cc] += hist[y, x, o]

        return hog_image

    # print(normalised_blocks)
    # print(normalised_blocks.shape)

    # from skimage.feature import hog
    # features = hog(srcimg, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=False)
    # print(features.shape)
    # features = features.reshape((126,126,9))

    return normalised_blocks
            
    # hog_image = visualise_histogram(features, 8, 8, signed_orientation=False)
    # plt.imshow(hog_image)
    # plt.show()

    """

    from skimage import feature
    import cv2
    import matplotlib.pyplot as plt
    image = cv2.imread('flower.jpg')

    (hog, hog_image) = feature.hog(image, orientations=9, 
                        pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                        block_norm='L2-Hys', visualize=True, transform_sqrt=True)

    cv2.imshow('HOG Image', hog_image)
    cv2.imwrite('hog_flower.jpg', hog_image*255.)
    cv2.waitKey(0)

    

    """