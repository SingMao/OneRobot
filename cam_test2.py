import cv2
import time
import os
import numpy as np

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 12800)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 7200)

cnt = 0
while os.path.isfile('calibrate/%d.png' % cnt):
    cnt += 1
path = 'calibrate/%d.png' % cnt

while True:
    ok, img = cap.read()
    # print(img.shape if ok else 'Q__Q')
    if not ok: continue

    _, maxval, _, maxpos = cv2.minMaxLoc(img[:,:,2])
    print(maxval, maxpos)
    # if maxval >= 150:
        # cv2.circle(img, maxpos, 20, (0, 255, 0), 5)
    # cv2.circle(img, (320, 240), 10, (255, 0, 0), 6)
    # cv2.circle(img, (400, 320), 10, (255, 0, 0), 10)

    cv2.imshow('test', img)
    cv2.imwrite(path, img)
    cv2.waitKey(10)
