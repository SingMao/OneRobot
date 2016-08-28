import cv2
import time
import os
import json
import numpy as np
import sys
from colorabc.alphabet import GROUPS, color_detect

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 1)
# print(cap.get(cv2.CAP_PROP_FPS))

# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)
# cap.set(cv2.CAP_PROP_CONTRAST, 0.1)
# cap.set(cv2.CAP_PROP_SATURATION, 0.1)
# cap.set(cv2.CAP_PROP_GAIN, 0.)
# cap.set(cv2.CAP_PROP_SHARPNESS, 0.2)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, False)

# os.system('v4l2-ctl -c exposure_absolute=20')
def edge_detect(img, mode, blur_size, erode_size, dilate_size):

    BLUR_SIZE = blur_size # 81 for large/19 for small 
    TRUNC_RATIO = 0.85
    ERODE_SIZE = erode_size
    DILATE_SIZE = dilate_size

    # denoised = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # too_bright=np.logical_and(img[:,:,1]<50, img[:,:,2]>200)
    # np.set_printoptions(threshold=np.nan)
    # np.savetxt('conconcon',img[:,:,1],'%i')
    # img[:,:,1]=np.where(too_bright, np.sqrt(img[:,:,1])+70, img[:,:,1])
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (BLUR_SIZE, BLUR_SIZE))

    edge = np.floor(0.5 * gray + 0.5 * (255 - blur)).astype('uint8')

    hist,bins = np.histogram(edge.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equ = cdf[edge]

    hist,bins = np.histogram(equ.flatten(),256,[0,256])
    max_idx = np.argmax(hist);
    hist_clean = np.where(equ > TRUNC_RATIO * max_idx, 255, equ)

    erode_kernel = np.ones((ERODE_SIZE, ERODE_SIZE), np.uint8)
    dilate_kernel = np.ones((DILATE_SIZE, DILATE_SIZE), np.uint8)
    if mode == 'alpha':
        dilation = cv2.dilate(hist_clean, dilate_kernel, iterations = 1)
        erosion = cv2.erode(dilation, erode_kernel, iterations = 1)
        closing = erosion
    else:
        erosion = cv2.erode(hist_clean, erode_kernel, iterations = 1)
        dilation = cv2.dilate(erosion, dilate_kernel, iterations = 1)
        closing = dilation
    # closing = cv2.morphologyEx(hist_clean, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(closing, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(100)
    binary = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('洪士號', binary)
    return binary

    # plt.imshow(binary, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(100)
    # exit()

def find_alphabet_contours(img):
    AREA_MIN_THRES = 0.005
    AREA_MAX_THRES = 0.15

    binary = edge_detect(img, mode='alpha', blur_size=19, erode_size=5, dilate_size=3)
    # plt.imshow(binary, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(100)

    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    real_contours = []
    # return contours
    for c in contours:
        area = cv2.contourArea(c) / (img.shape[0] * img.shape[1])
        if area > AREA_MIN_THRES and area < AREA_MAX_THRES:
            real_contours.append(c)
    return real_contours


with open('dclab_img/letter_cnt.json', 'r') as f:
    letter_cnt = json.loads(f.read())

radius = 108

group_id = -1

while True:
    try:
        ok, img = cap.read()
        # print(img.shape if ok else 'Q__Q')
        if not ok:
            print('Q__q')
            continue

        g = input('group_id ({}):'.format(group_id))
        if g:
            group_id = int(g)
        if group_id < 0:
            continue

        print(GROUPS[group_id])

        group = GROUPS[group_id]

        alpha_contours = find_alphabet_contours(img)
        windows = []
        for c in alpha_contours:
            center = np.reshape(np.mean(c, 0), 2).astype(int)
            # cv2.circle(img, (center[0], center[1]), 10, (255, 0, 0), 6)
            windows.append([center-radius, center+radius])
        for rec in windows:
            patch = img[int(rec[0][1]):int(rec[1][1]),int(rec[0][0]):int(rec[1][0]),:]
            lt = color_detect(patch, group_id)
            if lt is None:
                continue
            c = group[color_detect(patch)]
            # cv2.imshow('patch{}'.format(c), patch)
            # cv2.waitKey(10)
            letter_cnt[c] += 1
            cv2.imwrite('dclab_img/{}{}.jpg'.format(c, letter_cnt[c]),
                        patch)

        with open('dclab_img/letter_cnt.json', 'w') as f:
            f.write(json.dumps(letter_cnt))

    except KeyboardInterrupt:
        break

