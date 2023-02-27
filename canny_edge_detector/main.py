import canny_edge as ced
import matplotlib.image as mpimg
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = 0.299 * b + 0.587 * g + 0.114 * r

    return gray

if __name__=="__main__":

    path = os.getcwd()
    path = os.path.join(path, 'lena.png')
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512,512))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # img = rgb2gray(img)
    img = img2[...]

    detector = ced.cannyEdgeDetector(img)
    img_final = detector.detect()
    cv2.imshow("res", img_final)
    cv2.waitKey()

    img2= cv2.Canny(img2, 100, 150, L2gradient=True)
    cv2.imshow("res2", img2)
    cv2.waitKey()
    cv2.destroyAllWindows()