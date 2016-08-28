import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import subprocess
import os
import fcntl
import cam

CELL_WIDTH = 32
CELL_HEIGHT = 32
ROWS = 10
COLS = 10
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
        if area < 0.1 or area > 0.8 : continue
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

cap = cv2.VideoCapture(0)
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

torch_process = subprocess.Popen(
    ['th', 'luas/classify.lua', 'model_best.t7', 'sis.npy'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

imgpath = '20160724-222718'
imglist = [os.path.join(imgpath, x) for x in os.listdir(imgpath)]

cam.calibrate(cap)

cnt = 0
pers_transform = None
while cnt < 5:
    try:
        print('It %d' % cnt)
        # imgidx = cnt % len(imglist)
        # img = cv2.imread(imglist[imgidx])
        img = cap.read()[1]

        print(img.shape)

        cv2.imshow('original', img)
        cv2.waitKey(10)

        dst, pers_transform = normalize_image(img)
        if dst is None:
            continue
        cnt += 1

        means = np.mean(dst, (0, 1))
        std = np.std(dst, (0, 1))

        # dst = (dst - means) / std
        # print(np.mean(dst), np.std(dst))

        cv2.imshow('dst', dst)

        sis = split_image(dst, CELL_WIDTH, CELL_HEIGHT, COLS, ROWS)
        big_array = np.array(sis)[:,:,:,:,::-1].reshape(-1, CELL_WIDTH, CELL_HEIGHT, 3) / 255.
        np.save('sis.npy', big_array)

        torch_process.stdin.write('1\n')
        # torch_process.stdin.flush()
        out = torch_process.stdout.readline()
        out = [int(x) for x in out.strip().split()]

        if len(out) != 100: continue

        print(np.array(out).reshape((10, 10)))

        ok = 0
        for i in range(100):
            if out[i] == i//10:
                ok += 1
        print('%d %%' % ok)

        # plt.imshow(dst)
        # plt.show()
        # cv2.imshow('dst', dst)
    except KeyboardInterrupt:
        torch_process.terminate()
        break

print('Good')
print(pers_transform)
inv_pers_transform = np.linalg.inv(pers_transform)

def pTrans(s, m):
    x, y = s
    arr = np.array([[[x, y]]]).astype(float)
    return cv2.perspectiveTransform(arr, m)[0][0]

while True:
    i = np.random.randint(0, 9)
    j = np.random.randint(0, 9)
    x = i * 32 + 16
    y = j * 32 + 16
    pt = pTrans((x, y), inv_pers_transform)
    print(pt)
    xx, yy = cam.move_to(*pt)
    t0 = time.time()
    while time.time() < t0 + 0.5:
        ok, img = cap.read()
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), 2)
        cv2.circle(img, (int(xx), int(yy)), 5, (0, 255, 0), 2)
        cv2.imshow('test', img)
        cv2.waitKey(10)

