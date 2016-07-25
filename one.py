import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import json

CELL_WIDTH = 32
CELL_HEIGHT = 32
ROWS = 10
COLS = 10
NEW_WIDTH = CELL_WIDTH * COLS
NEW_HEIGHT = CELL_HEIGHT * ROWS

def normalize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(binary)
    # plt.show()
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 10)
    # plt.imshow(img)
    # plt.show()

    real_contour = None

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) / (img.shape[0] * img.shape[1])
        if area < 0.1 or area > 0.97 : continue
        real_contour = contours[i]

    epsilon = 0.1 * cv2.arcLength(real_contour, True)
    approx_contour = cv2.approxPolyDP(real_contour, epsilon, True)


    pers_src = np.float32([x[0] for x in approx_contour][:4])
    # print('Edges :', len(approx_contour))
    idx = np.argmin(pers_src[:,0] + pers_src[:,1])
    perm = list(range(idx, 4)) + list(range(0, idx))
    pers_src = pers_src[perm, :]
    pers_dst = np.float32([(0, 0), (NEW_WIDTH, 0), (NEW_WIDTH, NEW_HEIGHT), (0, NEW_HEIGHT)])

    pers_transform = cv2.getPerspectiveTransform(pers_src, pers_dst)

    dst = cv2.warpPerspective(img, pers_transform, (NEW_WIDTH, NEW_HEIGHT))
    return dst

def split_image(img, cw, ch, w, h):
    return [[img[i*cw:(i+1)*cw,j*ch:(j+1)*ch] for j in range(h)] for i in range(w)]

while True:
    img = cv2.imread('20160724-222718/one1.jpg')
    dst = normalize_image(img)
    sis = split_image(dst, CELL_WIDTH, CELL_HEIGHT, COLS, ROWS)
    big_array = np.array(sis)[:,:,:,:,::-1].reshape(-1, CELL_WIDTH, CELL_HEIGHT, 3)
    np.save('sis.npy', big_array / 255.)

    # plt.imshow(dst)
    # plt.show()
    cv2.imshow('test', dst)
    cv2.waitKey(100)

