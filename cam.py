import cv2
import time
import os
import pyuarm
import time
import numpy as np
from sk import ArmCalibrate

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

def get_maxpt(cap):
    ok, img = cap.read()
    # print(img.shape if ok else 'Q__Q')
    if not ok:
        return None

    _, maxval, _, maxpos = cv2.minMaxLoc(img[:,:,2])
    # print(maxval, maxpos)
    if maxval >= 140:
        cv2.circle(img, maxpos, 20, (0, 255, 0), 5)
    else:
        maxpos = None

    cv2.imshow('test', img)
    cv2.waitKey(10)
    return maxpos

uarm = pyuarm.uArm()
uarm.attach_all_servos()

uarm.write_servo_angle(pyuarm.SERVO_BOTTOM, 90, False)
uarm.write_servo_angle(pyuarm.SERVO_LEFT, 90, False)
uarm.write_servo_angle(pyuarm.SERVO_RIGHT, 30, False)

AC = ArmCalibrate()

def writeBR(b, r):
    uarm.write_servo_angle(pyuarm.SERVO_BOTTOM, b, False)
    uarm.write_servo_angle(pyuarm.SERVO_RIGHT, r, False)

def calibrate(cap):
    cap.read()

    os.system('v4l2-ctl -c exposure_auto=1')
    os.system('v4l2-ctl -c exposure_absolute=20')

    angles = []
    pts = []

    for i in range(10):
        phi = np.random.randint(75, 100)
        theta = np.random.randint(10, 35)
        # phi = 75 + i
        # theta = 10 + i
        writeBR(phi, theta)
        t0 = time.time()
        while time.time() < t0 + 1.:
            get_maxpt(cap)
            cv2.waitKey(10)
        ang = np.mean([np.array(uarm.read_servo_angle()[:3]) for x in range(4)], axis=0)
        maxpos = get_maxpt(cap)
        if maxpos is not None:
            angles.append(ang)
            pts.append(maxpos)

    angles = np.array(angles)
    pts = np.array(pts)
    # np.savez('angs10.npz', angles=angles, pts=pts)

    AC.fit(angles, pts)

    # for p, a in zip(pts, angles):
        # print(p, AC.predict(a))

    # for i in range(0, 601, 200):
        # for j in range(0, 401, 200):
            # phi, theta = AC.inv_predict((i, j))
            # writeBR(phi, theta)
            # t0 = time.time()
            # while time.time() < t0 + 1.:
                # get_maxpt(cap)
                # cv2.waitKey(10)
            # mpt = get_maxpt(cap)
            # print(i, j, mpt, phi, theta, AC.predict((phi, 0, theta)))
            # print(uarm.read_servo_angle()[:3], AC.predict(uarm.read_servo_angle()[:3]))

    os.system('v4l2-ctl -c exposure_auto=3')

    print('Calibration Completed')

def move_to(x, y):
    theta, phi = AC.inv_predict((x, y))
    writeBR(theta, phi)
    time.sleep(1.)
    return AC.predict(uarm.read_servo_angle()[:3])
