import cv2
import matplotlib.pyplot as plt
import numpy as np

NEW_WIDTH = 320
NEW_HEIGHT = 320

def normalize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    real_contour = None

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) / (img.shape[0] * img.shape[1])
        if area < 0.1 or area > 0.97 : continue
        print(i, area)
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

img = cv2.imread('20160724-222718/IMG_2393.JPG')

dst = normalize_image(img)

plt.imshow(dst)
plt.show()
