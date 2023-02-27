from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy import misc
import numpy as np
import cv2
import matplotlib.pyplot as plt

class cannyEdgeDetector:
    def __init__(self, img, sigma=1.4, weak_pixel=100, strong_pixel=255, lowthreshold=0.09, highthreshold=0.17):
        self.img = img
        self.img_final = None
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 
    
    def gaussian_kernel(self, sigma):
        # 영상 내 노이즈 제거해야함
        # 에지의 세기가 감소할 수 있으므로 적절한 표준편차 선택이 필요
        # 잡음이 심하지않으면 생략 가능
        
        # kernel_size = 5
        kernel_size = np.round(8*sigma+1) # get from hwang's CV book p.257
        print("gaussian_kernel mask size ", kernel_size)
        size = kernel_size // 2
        x, y = np.mgrid[-size:size+1, -size:size+1] # filter mask generation
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def sobel_filters(self, img):

        # 1x3 3x1 grad 크기만 고려한다면
        # 3x3 소벨은 좀 더 정확한 엣지를 찾기 위하여 그래디언트 방향도 함께 고려
        # 그러므로 가로방향 세로방향 소벨 마스크 필터링 수행한 후
        # 그래디언트 크기와 방향을 모두 계산해야 함 
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        # Ix = ndimage.filters.convolve(img, Kx)
        # Iy = ndimage.filters.convolve(img, Ky)
        Ix = cv2.filter2D(img, -1, Kx)
        Iy = cv2.filter2D(img, -1, Ky)

        # hypotenuse ( diagonal of right triangle)
        G = np.hypot(Ix, Iy) # Evaluate sqrt(Ix**2 + Iy**2), G size == img size
        # G = G / G.max() * 255 # TODO for absolute value (need to understand)
        theta = np.arctan2(Iy, Ix) # theta size == img size
        return (G, theta)
    
    # Set pixel edges from the pixels that are local maximum of gradient magnitude
    # Typically, find out the local maximum in 2D image
    # whether the specific pixel is local maximum, Investigate pixels surround the specific pixel.
    # => check (gradient direction of the pixel == gradient direction nearest pixels)
    # => Search for (pixel location of the most derivative)
    # If the magnitude of the current pixel is greater than the magnitudes of the neighbors, nothing changes, 
    # otherwise, the magnitude of the current pixel is set to zero.
    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        

        # angle = np.rad2deg(angle) % 180
        angle = np.rad2deg(D)
        angle[angle < 0] += 180 # the circle is symmetry

        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255
                    
                    """
                    find neighbour pixels to visit from the grad directions
                    q,r = two neigbours toward same direction 
                    """
                    # vertical
                    #angle 0    
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]

                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    
                    # horizontal
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    """
                    local maxima
                    """
                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        # suppressed
                        Z[i,j] = 0

                except IndexError as e:
                    pass
        
        # Z = nonmaxima suppressed image

        return Z

    def threshold(self, img):

        # highThreshold = 150
        highThreshold = img.max() * self.highThreshold
        # lowThreshold = 100
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        # define gray value of a WEAK and a STRONG pixel
        # for looking good
        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self, img):
        
        res_image = np.zeros_like(img)
        M, N = img.shape

        weak = self.weak_pixel
        strong = self.strong_pixel
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == strong): 
                    res_image[i,j] = strong
                    try:
                        if ((img[i+1, j-1] == weak) or (img[i+1, j] == weak) or (img[i+1, j+1] == weak)
                            or (img[i, j-1] == weak) or (img[i, j+1] == weak)
                            or (img[i-1, j-1] == weak) or (img[i-1, j] == weak) or (img[i-1, j+1] == weak)):
                            # if strong is nearby weak, weak edge is judged as a valid edge.
                            res_image[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return res_image
    
    def detect(self):
        self.img_smoothed = cv2.filter2D(self.img, -1, self.gaussian_kernel(self.sigma))
        self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
        self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
        self.thresholdImg = self.threshold(self.nonMaxImg)
        img_final = self.hysteresis(self.thresholdImg)
        self.img_final = np.array(img_final, dtype=np.uint8)
        
        return self.img_final   