import numpy as np
import cv2

def cam_to_global_coord(H, th, ph1, ph2, w, h, x, y):
    tanph22 = np.tan(ph2) * np.cos(ph2)
    pers_src = np.float32([ (0, 0), (w, 0), (w, h), (0, h) ])
    pers_dst = np.float32([
        (-H / np.cos(th + ph1) * tanph22, H * np.tan(th + ph1)),
        ( H / np.cos(th + ph1) * tanph22, H * np.tan(th + ph1)),
        ( H / np.cos(th - ph1) * tanph22, H * np.tan(th - ph1)),
        (-H / np.cos(th - ph1) * tanph22, H * np.tan(th - ph1)),
    ])

    transform = cv2.getPerspectiveTransform(pers_src, pers_dst)

    pos_cam = np.array([x, y, 1])
    pos_glo = np.dot(transform, pos_cam)
    real_pos = pos_glo[:-1] / pos_glo[-1]

    return real_pos

H = 37.8
th = np.arctan(28.8 / H)
ph1 = np.arctan(22 * (3/4) / 57.5)
ph2 = np.arctan(22 / 57.5)
w = 640
h = 480

x = 400
y = 320

print(cam_to_global_coord(H, th, ph1, ph2, w, h, x, y))
