import cv2
import time
import os
import pyuarm
import time
import numpy as np

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 12800)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 7200)
# cap.set(cv2.CAP_PROP_FPS, 1)
# print(cap.get(cv2.CAP_PROP_FPS))

# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)
# cap.set(cv2.CAP_PROP_CONTRAST, 0.1)
# cap.set(cv2.CAP_PROP_SATURATION, 0.1)
# cap.set(cv2.CAP_PROP_GAIN, 0.)
# cap.set(cv2.CAP_PROP_SHARPNESS, 0.2)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, False)

cap.read()

os.system('v4l2-ctl -c exposure_absolute=20')

def get_maxpt():
    ok, img = cap.read()
    # print(img.shape if ok else 'Q__Q')
    if not ok:
        return None

    _, maxval, _, maxpos = cv2.minMaxLoc(img[:,:,2])
    # print(maxval, maxpos)
    if maxval >= 140:
        cv2.circle(img, maxpos, 20, (0, 255, 0), 5)
    else:
        return None

    cv2.imshow('test', img)
    cv2.waitKey(10)
    return maxpos

uarm = pyuarm.uArm()
uarm.attach_all_servos()

uarm.write_servo_angle(pyuarm.SERVO_BOTTOM, 90, False)
uarm.write_servo_angle(pyuarm.SERVO_LEFT, 70, False)
uarm.write_servo_angle(pyuarm.SERVO_RIGHT, 30, False)

print(uarm.read_servo_angle()[:3])

angles = []
pts = []

for i in range(200):
    if i % 10 == 0:
        print(i)
    phi = np.random.randint(75, 105)
    theta = np.random.randint(10, 40)
    # phi = 75 + i
    # theta = 10 + i
    uarm.write_servo_angle(pyuarm.SERVO_BOTTOM, phi, False)
    uarm.write_servo_angle(pyuarm.SERVO_RIGHT, theta, False)
    t0 = time.time()
    while time.time() < t0 + 1.:
        get_maxpt()
        cv2.waitKey(10)
    ang = np.mean([np.array(uarm.read_servo_angle()[:3]) for x in range(4)], axis=0)
    maxpos = get_maxpt()
    if maxpos is not None:
        angles.append(ang)
        pts.append(maxpos)

angles = np.array(angles)
pts = np.array(pts)
np.savez('angs.npz', angles=angles, pts=pts)

print('Complete')
exit()

while True:
    maxpos = get_maxpt()

# while True:
    # ok, img = cap.read()
    # # print(img.shape if ok else 'Q__Q')
    # if not ok: continue

    # _, maxval, _, maxpos = cv2.minMaxLoc(img[:,:,2])
    # print(maxval, maxpos)
    # if maxval >= 150:
        # cv2.circle(img, maxpos, 20, (0, 255, 0), 5)

    # cv2.imshow('test', img)
    # cv2.waitKey(10)
