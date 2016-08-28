import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time

clr = np.array([[[148,153,146]]]).astype('uint8')
clr_list = clr.tolist()

def rotate(img, deg):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols//2,rows//2), deg, 1)
    dst = cv2.warpAffine(img, M, (cols,rows), borderMode=cv2.BORDER_REPLICATE)
    return dst

def rotate_scale(img, deg, rx, ry):
    r = rotate(img, deg)
    return cv2.resize(r, (0, 0), None, rx, ry)


tpl = cv2.imread('template/RA.png')
mask = (tpl[:,:,2] >= 100) & (tpl[:,:,0] <= 70) & (tpl[:,:,1] <= 70)
tpl = tpl * mask[:,:,np.newaxis]
tpl += clr * ~mask[:,:,np.newaxis]


lst = os.listdir('test')

for fpath in lst:
    if '.jpg' not in fpath or fpath[0] != 'a': continue

    img = cv2.imread(os.path.join('test', fpath))
    edge = cv2.Canny(img, 20, 100)
    R = 3
    aedge = cv2.GaussianBlur(edge, (3*R, 3*R), R)

    best_val = 1e20
    best_loc, bth, brx, bry = None, None, None, None

    t0 = time.time()
    for theta in np.linspace(0, 360, 12+1):
        for rx in np.linspace(0.05, 0.25, 10):
            for ry in np.linspace(0.05, 0.25, 10):
                tpl_r = rotate_scale(tpl, theta, rx, ry)
                mat = cv2.matchTemplate(img, tpl_r, cv2.TM_CCOEFF_NORMED)
                _, Val, _, Loc = cv2.minMaxLoc(mat)
                Val = -Val
                if Val < best_val:
                    best_val, best_loc, bth, brx, bry = Val, Loc, theta, rx, ry
                # print(theta, rx, ry, minVal)

    print(time.time() - t0)

    tpl_r = rotate_scale(tpl, bth, brx, bry)
    print(bth, brx, bry)
    (startX, startY) = (int(best_loc[0]), int(best_loc[1]))
    (endX, endY) = (int((best_loc[0] + tpl_r.shape[1])), int((best_loc[1] + tpl_r.shape[0])))
    cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 2)


    cv2.imshow('tpl', tpl_r)
    cv2.imshow('img', img)
    cv2.imshow('mat', mat)
    # cv2.imshow('edge', edge)
    # cv2.imshow('aedge', aedge)
    # cv2.waitKey(0)
    for i in range(100): cv2.waitKey(10)
