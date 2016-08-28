import numpy as np
import cv2


img = cv2.imread('../alphabet_data/test/a2.jpg')
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img[:,:,0], None)
kp2, des2 = sift.detectAndCompute(img[:,:,1], None)
kp3, des3 = sift.detectAndCompute(img[:,:,2], None)
lala = cv2.drawKeypoints(img, kp1, img, flags=4)
lala = cv2.drawKeypoints(lala, kp2, lala, flags=4)
lala = cv2.drawKeypoints(lala, kp3, lala, flags=4)
while True:
    cv2.imshow('A', lala)
    cv2.waitKey(10)
