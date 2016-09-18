import cv2
import numpy as np


img = cv2.imread('yz.jpg')
img = cv2.resize(img, (0, 0), None, 0.2, 0.2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = img.copy()
img2[:380,:,:] = 0

cv2.imshow('img', img2)
cv2.waitKey(0)

NR, NC = 5, 6

ret, bdr = cv2.findChessboardCorners(img2, (NR, NC))
cv2.drawChessboardCorners(img, (NR, NC), bdr, True)

UNIT = 0.02816 * 1280

objp = np.mgrid[0:NR,0:NC].T.reshape(-1,2) * 0.02816 * 1280
objp[:,0] += UNIT*5
objp[:,1] += UNIT*0
p_src = bdr[:,0,:].astype(np.float32)
p_dst = objp.astype(np.float32)
pers_transform, mask = cv2.findHomography(p_src, p_dst)

img = cv2.warpPerspective(img, pers_transform, (480, 300))
img = img.transpose([1, 0, 2])[:,::-1,:]

# corners = cv2.cornerHarris(gray, 2, 3, 0.04)
# corners = cv2.normalize(corners, None, 0, 1, cv2.NORM_MINMAX)
# print(cv2.minMaxLoc(corners))

for i in range(2):
    for j in range(3):
        x = img[int(UNIT*2*i):int(UNIT*2*(i+1)),img.shape[1]-int(UNIT*2*(j+1)):img.shape[1]-int(UNIT*2*j)]
        print(x.shape)
        cv2.imshow('x', x)
        cv2.imwrite('%d-%d.png' % (i, j), x)

cv2.imshow('img', img)
cv2.waitKey(0)
