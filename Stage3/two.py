import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import subprocess
import os
import fcntl

CELL_WIDTH = 32
CELL_HEIGHT = 32
ROWS = 16
COLS = 16
NEW_WIDTH = CELL_WIDTH * COLS
NEW_HEIGHT = CELL_HEIGHT * ROWS

def normalize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(binary)
    # plt.show()
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 10)
    # cv2.imshow('original', img)
    # cv2.waitKey(100)

    real_contour = None

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) / (img.shape[0] * img.shape[1])
        if area < 0.1 or area > 1.8 : continue
        real_contour = contours[i]
        epsilon = 0.05 * cv2.arcLength(real_contour, True)
        approx_contour = cv2.approxPolyDP(real_contour, epsilon, True)
        if len(approx_contour) == 4:
            break
        else:
            real_contour = None
    
    if real_contour is None:
        return None, None

    if len(approx_contour) != 4:
        return None, None

    # cv2.drawContours(img, [real_contour], 0, (255, 0, 0), 10)
    # cv2.imshow('cont', img)

    pers_src = np.float32([x[0] for x in approx_contour][:4])
    # print('Edges :', len(approx_contour))
    idx = np.argmin(pers_src[:,0] + pers_src[:,1])
    perm = list(range(idx, 4)) + list(range(0, idx))
    pers_src = pers_src[perm, :]
    pers_dst = np.float32([(0, 0), (NEW_WIDTH, 0), (NEW_WIDTH, NEW_HEIGHT), (0, NEW_HEIGHT)])

    pers_transform = cv2.getPerspectiveTransform(pers_src, pers_dst)

    dst = cv2.warpPerspective(img, pers_transform, (NEW_WIDTH, NEW_HEIGHT))
    return dst, pers_transform

def split_image(img, cw, ch, w, h):
    return [[img[i*cw:(i+1)*cw,j*ch:(j+1)*ch] for j in range(h)] for i in range(w)]

torch_process = subprocess.Popen(
    ['th', 'luas/classify.lua', 'model_best_binary_jitter_blur_weights2_ascii.t7', 'sis.npy'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

imgpath = 'img'
imglist = [os.path.join(imgpath, x) for x in os.listdir(imgpath)]

cnt = 0
pers_transform = None
while cnt < len(imglist):
    cnt += 1
    try:
        imgidx = cnt % len(imglist)
        img = cv2.imread(imglist[imgidx])
        print('It %d : %s' % (cnt, imglist[imgidx]))
        # img = cap.read()[1]

        print(img.shape)
        # cv2.imshow('img', img)

        dst, pers_transform = normalize_image(img)
        if dst is None:
            continue
        # dst = img
        print(dst.shape)

        means = np.mean(dst, (0, 1))
        std = np.std(dst, (0, 1))

        # dst = (dst - means) / std
        # print(np.mean(dst), np.std(dst))

        # cv2.imshow('dst', dst)

        sis = split_image(dst, CELL_WIDTH, CELL_HEIGHT, COLS, ROWS)
        big_array = np.array(sis)[:,:,:,:,::-1].reshape(-1, CELL_WIDTH, CELL_HEIGHT, 3) / 255.
        np.save('sis.npy', big_array)

        torch_process.stdin.write('1\n')
        # torch_process.stdin.flush()
        out = torch_process.stdout.readline()
        print(out)
        out = [int(x) for x in out.strip().split()]

        if len(out) != ROWS*COLS: continue

        nout = np.array(out).reshape((ROWS, COLS))
        ngood = ((nout==1) + (nout==3) + (nout==9)) * 1
        # print(ngood)
        print(nout)

        ncorrect = np.zeros(nout.shape)
        ncorrect[1, 9] = 1
        ncorrect[4, 3] = 1
        ncorrect[4, 10] = 1
        ncorrect[4, 12] = 1
        ncorrect[7, 9] = 1
        ncorrect[8, 2] = 1
        ncorrect[10, 2]  =1
        ncorrect[12, 13] = 1
        ncorrect[13, 3] = 1
        ncorrect[15, 0] = 1

        acc = float((ncorrect == nout).sum()) / (ROWS * COLS)
        print('%d %%' % (acc * 100))
        print('Positive predictions: %d' % nout.sum())
        print('True positives: %d' % np.logical_and(nout, ncorrect).sum())
        print('False positives: %d' % np.logical_and(nout, np.logical_not(ncorrect)).sum())
        print('False negatives: %d' % np.logical_and(ncorrect, np.logical_not(nout)).sum())

        # ok = 0
        # for i in range(ROWS*COLS):
            # if out[i] == i//ROWS:
                # ok += 1
        # print('%d %%' % ok)

        # plt.imshow(dst)
        # plt.show()
        # cv2.imshow('dst', dst)
    except KeyboardInterrupt:
        torch_process.terminate()
        break

torch_process.terminate()
exit()

print('Good')
print(pers_transform)
inv_pers_transform = np.linalg.inv(pers_transform)

def pTrans(s, m):
    x, y = s
    arr = np.array([[[x, y]]]).astype(float)
    return cv2.perspectiveTransform(arr, m)[0][0]

