import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

def img2huevec(iim):
    hue = cv2.cvtColor(iim, cv2.COLOR_BGR2HSV)[:,:,0] / 180. * 2. * np.pi
    hvec = np.concatenate([
        np.cos(hue)[:,:,np.newaxis],
        np.sin(hue)[:,:,np.newaxis],
    ], axis=2)
    return hvec.astype(np.float32)

path = sys.argv[1] if len(sys.argv) >= 2 else 'da.png'

cap = cv2.VideoCapture(0)

# mtx = np.load('mtx.npy')
# dist = np.load('dist.npy')

tpl = cv2.imread('tpl.png')
tpl = cv2.resize(tpl, (0, 0), None, 0.15, 0.15)
tpl_mask = (tpl[:,:,0] < 255).astype('float32')
orig_tpl = tpl

shrink_ratio = 0.3

tpl = cv2.resize(tpl, (0, 0), None, shrink_ratio, shrink_ratio)
tpl = cv2.Canny(tpl, 100, 200).astype(np.float32)
R = 1
tpl = cv2.GaussianBlur(tpl, (3*R, 3*R), R) / 40.
tpl = np.minimum(tpl, 1) - 0.8
tpl_mask = cv2.resize(tpl_mask, (0, 0), None, shrink_ratio, shrink_ratio)
orig_tpl = cv2.resize(orig_tpl, (0, 0), None, shrink_ratio, shrink_ratio)
tpl = np.maximum(tpl, -tpl_mask)
tpl_hue = img2huevec(orig_tpl) * tpl_mask[:,:,np.newaxis]

# plt.imshow(tpl)
# plt.show()

if False:
    ok, img = cap.read()
    cv2.imwrite('hh.png', img)

cal = cv2.imread('hh.png')
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2) * 0.02816 * 1280 + 150

ret, bdr = cv2.findChessboardCorners(cal, (6, 9))
# ret, rvec, tvec = cv2.solvePnP(objp, bdr, mtx, dist)

bdr = np.array(bdr)
p_src = bdr[:,0,:].astype(np.float32)
p_dst = objp[:,:2].astype(np.float32)
pers_transform, mask = cv2.findHomography(p_src, p_dst)
print(pers_transform)

while True:
    ok = False
    while not ok:
        ok, img = cap.read()
    # img = cv2.imread(path)

    # dst = cv2.warpPerspective(cal, pers_transform, (480, 540))
    img2 = cv2.warpPerspective(img, pers_transform, (480, 600))
    # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orig_img2 = img2

    # img2 = cv2.imread('test/a1.jpg')
    # img2 = img2[30:150,230:350,:]

    def get_sift_kp_feat(iim):
        sift = cv2.xfeatures2d.SIFT_create()
        kps = []
        feats = []
        for c in range(3):
            kp = sift.detect(iim[:,:,c], None)
            kps.extend(kp)

        for c in range(3):
            kp2, feat = sift.compute(iim[:,:,c], kps)
            feats.append(feat)
        feats = np.concatenate(feats, axis=1)
        return kps, feats.astype(np.float32)

    def get_sift_kp_feat_gray(iim):
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(iim, None)
        kp2, feat = sift.compute(iim, kp)
        return kp, feat.astype(np.float32)


    img2 = cv2.resize(img2, (0, 0), None, shrink_ratio, shrink_ratio)
    img2 = cv2.Canny(img2, 100, 200).astype(np.float32)
    img2 /= np.max(img2)

    orig_img2 = cv2.resize(orig_img2, (0, 0), None, shrink_ratio, shrink_ratio)
    img2_hue = img2huevec(orig_img2)


    def rotate(img, deg):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols//2,rows//2), deg, 1)
        dst = cv2.warpAffine(img, M, (cols,rows), borderMode=cv2.BORDER_REPLICATE)
        return dst

    def rotate_scale(img, deg, rx, ry):
        r = rotate(img, deg)
        return cv2.resize(r, (0, 0), None, rx, ry)

    best_val = 1e20
    best_loc, bth, brx, bry = None, None, None, None
    t0 = time.time()
    th_range = (0, 360)
    mp = np.ones(img2.shape[:2]) * 0
    for rnd in range(2):
        for theta in np.linspace(th_range[0], th_range[1], 36+1):
            for rx in np.linspace(0.69, 0.69, 1):
                for ry in np.linspace(0.69, 0.69, 1):
                    tpl_r = rotate_scale(tpl, theta, rx, ry)
                    tpl_r_hue = rotate_scale(tpl_hue, theta, rx, ry)
                    mat = cv2.matchTemplate(img2, tpl_r, cv2.TM_CCORR_NORMED)
                    # cv2.imshow('aa', img2_hue[:,:,0])
                    # cv2.imshow('bb', img2_hue[:,:,1])
                    # cv2.imshow('cc', tpl_r_hue[:,:,0])
                    # cv2.imshow('dd', tpl_r_hue[:,:,1])
                    # cv2.waitKey(0)
                    total_sz = tpl_r_hue.shape[0] * tpl_r_hue.shape[1]
                    mat += (total_sz-cv2.matchTemplate(img2_hue, tpl_r_hue, cv2.TM_SQDIFF)) * 0.0003
                    a, b = mat.shape
                    mp[:a,:b] = np.maximum(mp[:a,:b], mat)
                    _, Val, _, Loc = cv2.minMaxLoc(mat)
                    Val = -Val
                    if Val < best_val:
                        best_val, best_loc, bth, brx, bry = Val, Loc, theta, rx, ry
        th_range = (bth-5, bth+5)

    print('Time : %.2f, Val : %.4f' % (time.time() - t0, best_val))

    tpl_r = rotate_scale(tpl, bth, brx, bry)
    tpl_hue_r = rotate_scale(tpl_hue, bth, brx, bry)
    orig_tpl_r = rotate_scale(orig_tpl, bth, brx, bry)
    print(bth, brx, bry)
    (startX, startY) = (int(best_loc[0]), int(best_loc[1]))
    (endX, endY) = (int((best_loc[0] + tpl_r.shape[1])), int((best_loc[1] + tpl_r.shape[0])))
    cv2.rectangle(img2, (startX, startY), (endX, endY), (255, 0, 0), 2)
    cv2.rectangle(orig_img2, (startX, startY), (endX, endY), (255, 0, 0), 2)
    cv2.rectangle(img2_hue, (startX, startY), (endX, endY), (255, 0, 0), 2)

    ###

    mp = cv2.normalize(mp, None, 0., 1., cv2.NORM_MINMAX)
    print(cv2.minMaxLoc(mp))

    cv2.imshow('mp', mp)
    cv2.imshow('tpl_r', tpl_r)
    cv2.imshow('orig_tpl_r', orig_tpl_r)
    # cv2.imshow('tpl_hue', tpl_hue)
    cv2.imshow('img2', img2)
    cv2.imshow('orig_img2', orig_img2)
    # cv2.imshow('img3', img3)
    # cv2.imshow('img2_hue0', img2_hue[:,:,0])
    # cv2.imshow('img2_hue1', img2_hue[:,:,1])
    # cv2.imshow('tpl_r_hue0', tpl_r_hue[:,:,0])
    # cv2.imshow('tpl_r_hue1', tpl_r_hue[:,:,1])

    cv2.waitKey(50)
