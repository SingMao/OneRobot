import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

CELL_WIDTH = 32
CELL_HEIGHT = 32
ROWS = 10
COLS = 10
NEW_WIDTH = CELL_WIDTH * COLS
NEW_HEIGHT = CELL_HEIGHT * ROWS

def normalize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    real_contour = None

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) / (img.shape[0] * img.shape[1])
        if area < 0.1 or area > 0.97 : continue
        real_contour = contours[i]


    epsilon = 0.1 * cv2.arcLength(real_contour, True)
    approx_contour = cv2.approxPolyDP(real_contour, epsilon, True)

    # cv2.drawContours(img, [approx_contour], 0, rand_color(), 10)

    pers_src = np.float32([x[0] for x in approx_contour][:4])
    idx = np.argmin(pers_src[:,0] + pers_src[:,1])
    perm = list(range(idx, 4)) + list(range(0, idx))
    pers_src = pers_src[perm, :]
    pers_dst = np.float32([(0, 0), (NEW_WIDTH, 0), (NEW_WIDTH, NEW_HEIGHT), (0, NEW_HEIGHT)])

    pers_transform = cv2.getPerspectiveTransform(pers_src, pers_dst)

    dst = cv2.warpPerspective(img, pers_transform, (NEW_WIDTH, NEW_HEIGHT))
    return dst

def split_image(img, cw, ch, w, h):
    return [[img[i*cw:(i+1)*cw,j*ch:(j+1)*ch] for j in range(h)] for i in range(w)]

img = cv2.imread('20160724-222718/IMG_2396.JPG')

t0 = time.time()
dst = normalize_image(img)
print(time.time()-t0)

sis = split_image(dst, CELL_WIDTH, CELL_HEIGHT, COLS, ROWS)

for i in range(COLS):
    for j in range(ROWS):
        #plt.imshow(sis[i][j])
        cv2.imwrite('small/test%d%d.png' % (i, j), sis[i][j])
        #cv2.waitKey(500)

#plt.imshow(sis[0][0])
#plt.show()
