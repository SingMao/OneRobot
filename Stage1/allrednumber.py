import cv2
import numpy as np

img = cv2.imread('redgreen/green1.jpg', cv2.CV_LOAD_IMAGE_UNCHANGED)
blah = cv2.Canny(img, 100, 200).astype(np.float32)

while True:
    cv2.imshow('ff', blah)
    cv2.waitKey(100)
