import cv2
import numpy as np
import glob

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2) * 0.02816

objpoints = []
imgpoints = []

ilists = glob.glob('calibrate/*.png')

for path in ilists:
    img = cv2.imread(path)
    shp = (img.shape[1], img.shape[0])

    ret, bdr = cv2.findChessboardCorners(img, (6, 9))
    if not ret: continue

    cv2.drawChessboardCorners(img, (6, 9), bdr, True)
    imgpoints.append(bdr)
    objpoints.append(objp)

    # cv2.imshow('img', img)
    # cv2.waitKey(500)
    # break

# print(objpoints)
# print(imgpoints)

rat = 640*57.5/22

init_mtx = np.array([
    [rat, 0, 320],
    [0, rat, 240],
    [0, 0, 1],
]).astype(np.float32)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, shp, init_mtx, None)#, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

print(mtx)
print(dist)

clr = np.array([[[255,255,255]]]).astype('uint8')
clr_list = clr.tolist()

tpl = cv2.imread('template/RA.png')
mask = (tpl[:,:,2] >= 100) & (tpl[:,:,0] <= 70) & (tpl[:,:,1] <= 70)
tpl = tpl * mask[:,:,np.newaxis]
tpl += clr * ~mask[:,:,np.newaxis]

img = cv2.imread('hh.png')
ret, bdr = cv2.findChessboardCorners(img, (6, 9))
ret, rvec, tvec = cv2.solvePnP(objp, bdr, mtx, dist)
print(rvec, tvec)

cv2.imshow('tpl', tpl)
cv2.imwrite('tpl.png', tpl)
cv2.imshow('img', img)
cv2.waitKey(0)
