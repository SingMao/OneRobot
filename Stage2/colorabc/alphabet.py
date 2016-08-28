#! /usr/bin/env python3

import matplotlib.pyplot as plt
from string import ascii_lowercase
import numpy as np
import cv2

# red
# orange
# yellow
# light green
# green
# light blue
# blue
# purple

LETTERS = [
    'aiqy',
    'bjrz',
    'cks',
    'dlt',
    'emu',
    'fnv',
    'gow',
    'hpx',
]

GROUPS = [
    'abcdefgh',
    'ijklmnop',
    'qrstuvwx',
    'yz',
]

# original hls table
# HLS: 0 ~ 180, 0 ~ 255, 0 ~ 255
# [180, 147, 180]
# [ 17, 151, 228]
# [ 25, 164, 241]
# [ 32, 143, 160]
# [ 37, 147, 140]
# [ 99, 197, 202]
# [100, 150, 168]
# [158, 192, 101]

TABLE = np.array([
    [[170,  20,  20], [ 12, 220, 255]],
    [[ 17,  40,  50], [ 25, 220, 255]],
    [[ 25,  20,  30], [ 33, 200, 255]],
    [[ 33,  20,  30], [ 38, 180, 255]],
    [[ 38,  50,  40], [ 45, 180, 255]],
    [[ 90, 100,  20], [110, 250, 200]],
    [[ 90,  50,  70], [110, 200, 255]],
    [[150,  20,  20], [170, 255, 150]],
], dtype=int)

COLORS = np.mean(TABLE, 1)
COLORS[:,0] = np.where(np.abs(COLORS[:,0] - TABLE[:,0,0]) < 45,
                       COLORS[:,0], (COLORS[:,0] + 90) % 180)
print(COLORS)
TOLS = np.abs(TABLE[:,1] - COLORS)
TOLS[:,0] = np.where(TOLS[:,0] < 45, TOLS[:,0], 90 - TOLS[:,0])
print(TOLS)

def color_detect(img, group_id=0):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if hls is None:
        return None
    h = hls[:,:,0].astype(int)
    l = hls[:,:,1].astype(int)
    s = hls[:,:,2].astype(int)

    coverages = []
    for c, tol, letter in zip(COLORS, TOLS, GROUPS[group_id]):
        hmask = np.amin([np.abs(h - c[0]), 180 - np.abs(h - c[0])], axis=0)
        hmask = hmask < tol[0]
        lmask = np.abs(l - c[1]) < tol[1]
        smask = np.abs(s - c[2]) < tol[2]

        mask = hmask & smask & lmask
        # plt.imshow(mask.astype('uint8'), cmap='Greys_r')
        # plt.show()
        # plt.clf()
        coverages.append(np.sum(mask.astype(int)))

    print(np.argmax(coverages), end='\t')
    print(np.array(coverages))
    return np.argmax(coverages)

# img = cv2.imread('../alphabet_data/test/{}.jpg'.format('a1'))
# print(color_detect(img))
# for c in ascii_lowercase:
    # img = cv2.imread('../alphabet_data/test/{}1.jpg'.format(c))
    # print(c, end='\t')
    # color_detect(img)
