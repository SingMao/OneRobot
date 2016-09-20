import cv2
import time
import os
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 960)
# cap.set(cv2.CAP_PROP_FPS, 1)
# print(cap.get(cv2.CAP_PROP_FPS))

# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)
# cap.set(cv2.CAP_PROP_CONTRAST, 0.1)
# cap.set(cv2.CAP_PROP_SATURATION, 0.1)
# cap.set(cv2.CAP_PROP_GAIN, 0.)
# cap.set(cv2.CAP_PROP_SHARPNESS, 0.2)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, False)

os.system('v4l2-ctl -c exposure_auto=1')
os.system('v4l2-ctl -c exposure_auto_priority=0')
os.system('v4l2-ctl -c exposure_absolute=20')

while True:
    ok, img = cap.read()
    # print(img.shape if ok else 'Q__Q')
    if not ok: continue

    # _, maxval, _, maxpos = cv2.minMaxLoc(img[:,:,2])
    # print(maxval, maxpos)
    # if maxval >= 150:
        # cv2.circle(img, maxpos, 20, (0, 255, 0), 5)
    # cv2.circle(img, (320, 240), 10, (255, 0, 0), 6)
    # cv2.circle(img, (400, 320), 10, (255, 0, 0), 10)

    cv2.imshow('test', img)
    # cv2.imwrite('ghijkl.png', img)
    cv2.waitKey(10)
