import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_mask(img):
    # HSV: 0 ~ 180, 0 ~ 255, 0 ~ 255
    white = np.array((0, 0, 255))
    tol = np.array((180, 25, 100))

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h = hls[:,:,0].astype(int)
    l = hls[:,:,1].astype(int)
    s = hls[:,:,2].astype(int)

    all_mask = (255 - l < 120) & (~(l < 50)) & (~((np.abs(h - 50) < 30) & (s > 125)))
    all_mask = all_mask.astype(int)

    return all_mask

def get_pers_trans():
    cal = cv2.imread('calib.jpg')
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2) * 0.02816 * 1280
    objp[:,0] += 360
    objp[:,1] += 500

    ret, bdr = cv2.findChessboardCorners(cal, (6, 9))

    bdr = np.array(bdr)
    p_src = bdr[:,0,:].astype(np.float32)
    p_dst = objp[:,:2].astype(np.float32)
    pers_transform, mask = cv2.findHomography(p_src, p_dst)

    return pers_transform, mask

# _, img = cap.read()
# cv2.imwrite('calib.jpg', img)

cap = None
pers_t, pers_mask = None, None

width = 960
height = 960
height_cut = 0
# center_poly = None
center_pos = None
max_area = 0

def get_bird_view(img):
    return cv2.warpPerspective(img, pers_t, (width, height))[height_cut:]

def get_img():
    ok = False
    while not ok:
        ok, img = cap.read()

    return get_bird_view(img)

def init(camid):
    global cap
    global pers_t, pers_mask
    global height
    global height_cut
    # global center_poly
    global center_pos
    global max_area
    cap = cv2.VideoCapture(camid)
    os.system('v4l2-ctl -d /dev/video1 -c white_balance_temperature_auto=0')
    os.system('v4l2-ctl -d /dev/video1 -c exposure_auto=1')
    os.system('v4l2-ctl -d /dev/video1 -c exposure_auto_priority=0')
    pers_t, pers_mask = get_pers_trans()

    ok = False
    while not ok:
        ok, img = cap.read()

    testimg = get_bird_view(np.ones(img.shape[:2], dtype=np.uint8))
    height = np.nonzero(np.sum(testimg, axis=1) > 200)[0][-1]
    height_cut = height - 100

    testimg = get_bird_view(np.ones(img.shape[:2], dtype=np.uint8))

    xavg = (np.sum(testimg * np.arange(testimg.shape[1]), axis=1) /
            (np.sum(testimg, axis=1) + 0.001))
    # center_poly = np.vstack((xavg.astype(int), np.arange(testimg.shape[0]))).T
    center_pos = int(np.mean(xavg))
    max_area = np.sum(testimg, axis=1)


def check_road(mask):
    area = np.sum(mask, axis=1)
    xavg = (np.sum(mask * np.arange(mask.shape[1]), axis=1) /
            (np.sum(mask, axis=1) + 0.0001))
    avg_pos = int(np.sum(xavg) / (np.count_nonzero(xavg) + 0.00001))

    if np.all(area / max_area > 0.1):
        return avg_pos - center_pos
    else:
        return None


if __name__ == '__main__':
    init(1)
    while True:
        try:
            img = get_img()
            mask = color_mask(img)
            pos = check_road(mask)

            # cv2.polylines(img, [center_poly], False, (0, 255, 0), 5)
            # cv2.polylines(img, [pos], False, (255, 0, 0), 5)
            cv2.circle(img, (center_pos, 50), 4, (0, 255, 0), 10)
            if pos:
                cv2.circle(img, (pos + center_pos, 50), 4, (255, 0, 0), 10)
            cv2.imshow('origin', img)
            cv2.imshow('mask', mask.astype(np.float32))
            cv2.waitKey(10)
        except KeyboardInterrupt:
            break
