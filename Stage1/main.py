from __future__ import print_function
import os
import cv2
import numpy as np

import camera
import car

def check_road():
    img = camera.get_img()
    mask = camera.color_mask(img)
    pos = camera.check_road(mask)

    cv2.circle(img, (camera.center_pos, 50), 4, (0, 255, 0), 10)
    if pos:
        cv2.circle(img, (pos + camera.center_pos, 50), 4, (255, 0, 0), 10)
    cv2.imshow('origin', img)
    # cv2.imshow('mask', mask.astype(np.float32))
    cv2.waitKey(10)

    if pos is None:
        return None
    elif abs(pos) < 30:
        return (0, 5, 0)
    else:
        print(pos, np.arctan(-pos / 600.))
        return (0, 0, np.arctan(-pos / 600.))


def _find_road(step, limit):
    degree = 0
    scale = np.pi / 180.
    while abs(degree) < abs(limit):
        degree += step
        car.rotate(step * scale)
        ret = check_road()
        if ret:
            car.move(0, -10)
            ret = check_road()
            car.move(0, 10)
        if ret:
            return True
    return False


def find_road():
    scale = np.pi / 180.
    limit = 100
    step = 5
    has_road = _find_road(step, limit)
    if not has_road:
        print("ASDFASDFASDF")
        car.rotate(-limit * scale)
        print("EEEEEEEE")
        has_road = _find_road(-step, -limit)
    return has_road


def forward(vec):
    if vec[2]:
        car.rotate(vec[2])
    else:
        car.move(0, vec[1])

def setup():
    # calibration()
    camera.init(1)
    car.init('/dev/ttyACM0')

def run():
    while True:
        ret = check_road()
        if ret is None:
            has_road = find_road()
            if not has_road:
                break
        else:
            print('gogo:', ret)
            forward(ret)

    print('EEE_E_EEE')

if __name__ == '__main__':
    setup()
    try:
        run()
    except KeyboardInterrupt:
        car.stop()

    car.turn_off()
