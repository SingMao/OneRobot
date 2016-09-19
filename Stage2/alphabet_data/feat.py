import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

HUE_R = 2.

def img2huevec(iim):
    hsv = cv2.cvtColor(iim, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0] / 180. * 2. * np.pi
    sat = hsv[:,:,1] / 255.
    hvec = np.concatenate([
        (np.cos(hue) * sat)[:,:,np.newaxis],
        (np.sin(hue) * sat)[:,:,np.newaxis],
        (np.sqrt(HUE_R*HUE_R-sat*sat))[:,:,np.newaxis],
    ], axis=2)
    return hvec.astype(np.float32)

def rotate(img, deg):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols//2,rows//2), deg, 1)
    dst = cv2.warpAffine(img, M, (cols,rows), borderMode=cv2.BORDER_REPLICATE)
    return dst

def rotate_scale(img, deg, rx, ry):
    r = rotate(img, deg)
    return cv2.resize(r, (0, 0), None, rx, ry)

def process_template(tpl):
    rgb = tpl[:,:,:3]
    alpha = tpl[:,:,3]

    tpl_mask = (alpha != 0).astype(np.uint8)

    num_good_pixels = int(np.sum(tpl_mask))

    rgb *= tpl_mask[:,:,np.newaxis]

    avg_rgb = np.mean(rgb, axis=(0, 1)) * (tpl.shape[0] * tpl.shape[1]) / num_good_pixels
    newrgb = tpl_mask[:,:,np.newaxis] * avg_rgb

    return newrgb.astype(np.uint8), tpl_mask.astype(np.float32)

path = sys.argv[1] if len(sys.argv) >= 2 else 'da.png'

cap = cv2.VideoCapture(0)

shrink_ratio = 0.5

tpls = []
tpls_mini = []

alphabet = 'abcdef'

for i, c in enumerate(alphabet):
    tpl = cv2.imread('tpl_%s.png' % c, cv2.CV_LOAD_IMAGE_UNCHANGED)
    tpl, tpl_mask = process_template(tpl)
    orig_tpl = tpl

    tpls.append([]) #(r, theta, tpl, tpl_hue, otpl)
    tpls_mini.append([])


    for r in np.linspace(0.45, 0.45, 1):
        for theta in np.linspace(0, 360, 36+1)[:-1]:
            _tpl = rotate_scale(tpl, theta, r, r)
            _otpl = rotate_scale(orig_tpl, theta, r, r)
            _mask = rotate_scale(tpl_mask, theta, r, r)
            _tpl = cv2.Canny(_tpl, 100, 200).astype(np.float32)
            R = 1
            _tpl = cv2.GaussianBlur(_tpl, (3*R, 3*R), R) / 40.
            _tpl = np.minimum(_tpl, 1) - 0.8
            _tpl = np.maximum(_tpl, -_mask)
            _hue = img2huevec(_otpl) * _mask[:,:,np.newaxis]

            tpls[i].append((r, theta, _tpl, _hue, _otpl))

            _tpl = cv2.resize(_tpl, (0, 0), None, shrink_ratio, shrink_ratio)
            _otpl = cv2.resize(_otpl, (0, 0), None, shrink_ratio, shrink_ratio)
            _hue = cv2.resize(_hue, (0, 0), None, shrink_ratio, shrink_ratio)

            tpls_mini[i].append((r, theta, _tpl, _hue, _otpl))

# exit(0)

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


for cnt in xrange(10**9):
    ok = False
    while not ok:
        ok, img = cap.read()
    # img = cv2.imread(path)
    
    idx = cnt % len(alphabet)

    # dst = cv2.warpPerspective(cal, pers_transform, (480, 540))
    img2 = cv2.warpPerspective(img, pers_transform, (480, 600))
    # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (0, 0), None, 0.5, 0.5)
    img2_canny = cv2.Canny(img2, 100, 200).astype(np.float32)
    img2_canny /= np.max(img2_canny)
    img2_hue = img2huevec(img2)

    img2_mini = cv2.resize(img2, (0, 0), None, shrink_ratio, shrink_ratio)
    img2_mini_canny = cv2.Canny(img2_mini, 100, 200).astype(np.float32)
    img2_mini_canny /= np.max(img2_mini_canny)
    img2_mini_hue = img2huevec(img2_mini)

    # cv2.imshow('img2', img2)
    cv2.imshow('img2_canny', img2_canny)
    # cv2.waitKey(10)

    candidates = [] #(val, r, th)
    t0 = time.time()

    for r, th, _tpl, _hue, _otpl in tpls_mini[idx]:
        mat = cv2.matchTemplate(img2_mini_canny, _tpl, cv2.TM_CCORR_NORMED)
        total_sz = _hue.shape[0] * _hue.shape[1]
        nzcnt1 = np.sum(np.sum(_otpl, 2) != 0)
        nzcnt = np.sum(_hue*_hue) / (HUE_R * HUE_R)
        # print(th, nzcnt1, nzcnt, total_sz)
        mat += (((total_sz-nzcnt) * HUE_R * HUE_R) - cv2.matchTemplate(img2_mini_hue, _hue, cv2.TM_SQDIFF)) / total_sz * 0.3
        a, b = mat.shape
        _, Val, _, Loc = cv2.minMaxLoc(mat)
        Val = -Val
        candidates.append((Val, r, th))
    candidates.sort()
    print(len(candidates))
    candidates = candidates[:5]
    candidates_set = set()
    for val, r, th in candidates:
        candidates_set.add((r, th))

    for x in candidates: print(x)

    print('Stage 1 Time : %.2f, Val : %.4f' % (time.time() - t0, candidates[0][0]))

    best_val = 1e20
    best_loc, bth, br = None, None, None
    btpl, botpl = None, None
    t0 = time.time()
    mp = np.ones(img2.shape[:2]) * (-1)

    for r, th, _tpl, _hue, _otpl in tpls[idx]:
        if (r, th) not in candidates_set: continue
        mat = cv2.matchTemplate(img2_canny, _tpl, cv2.TM_CCORR_NORMED)
        total_sz = _hue.shape[0] * _hue.shape[1]
        nzcnt = np.sum(np.sum(_otpl, 2) != 0)
        nzcnt = np.sum(_hue*_hue) / (HUE_R * HUE_R)
        cv2.imshow('ttt', np.sum(_otpl, 2).astype(np.float32))
        cv2.waitKey(10)
        mat += (((total_sz-nzcnt) * HUE_R * HUE_R) - cv2.matchTemplate(img2_hue, _hue, cv2.TM_SQDIFF)) / total_sz * 0.3
        # print(total_sz)
        a, b = mat.shape
        mp[:a,:b] = np.maximum(mp[:a,:b], mat)
        _, Val, _, Loc = cv2.minMaxLoc(mat)
        Val = -Val
        if Val < best_val:
            best_val, best_loc, bth, br = Val, Loc, th, r
            btpl, botpl = _tpl, _otpl


    print('Stage 2 Time : %.2f, Val : %.4f' % (time.time() - t0, best_val))

    # tpl_r = rotate_scale(tpl, bth, brx, bry)
    # tpl_hue_r = rotate_scale(tpl_hue, bth, brx, bry)
    # orig_tpl_r = rotate_scale(orig_tpl, bth, brx, bry)
    print(bth, br)
    (startX, startY) = (int(best_loc[0]), int(best_loc[1]))
    (endX, endY) = (int((best_loc[0] + btpl.shape[1])), int((best_loc[1] + btpl.shape[0])))
    cv2.rectangle(img2, (startX, startY), (endX, endY), (255, 0, 0), 2)
    # cv2.rectangle(orig_img2, (startX, startY), (endX, endY), (255, 0, 0), 2)
    # cv2.rectangle(img2_hue, (startX, startY), (endX, endY), (255, 0, 0), 2)

    ###

    real_min = np.min((mp != -1) * mp)
    mp = np.maximum(mp, real_min)

    mp = cv2.normalize(mp, None, 0., 1., cv2.NORM_MINMAX)
    mp = (mp * 255).astype(np.uint8)
    # print(cv2.minMaxLoc(mp))
    mpjet = cv2.applyColorMap(mp, cv2.COLORMAP_JET)
    print(mpjet.shape)

    while True:
        cv2.imshow('mp', mpjet)
        # cv2.imshow('tpl_r', tpl_r)
        cv2.imshow('tpl', btpl)
        cv2.imshow('otpl', botpl)
        # cv2.imshow('tpl_hue', tpl_hue)
        cv2.imshow('img2', img2)
        # cv2.imshow('orig_img2', orig_img2)
        # cv2.imshow('img3', img3)
        cv2.imshow('img2_hue0', (img2_hue[:,:,0]+1)/2)
        cv2.imshow('img2_hue1', (img2_hue[:,:,1]+1)/2)
        # cv2.imshow('tpl_r_hue0', tpl_r_hue[:,:,0])
        # cv2.imshow('tpl_r_hue1', tpl_r_hue[:,:,1])

        cv2.waitKey(50)
        break
