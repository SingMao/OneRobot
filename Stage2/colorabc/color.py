import numpy as np
import cv2

# np.set_printoptions(threshold=np.inf)

# colors = [
    # (223,  70,  72),
    # (244, 166,  57),
    # (250, 219,  77),
    # (202, 213,  72),
    # (177, 206,  87),
    # (151, 215, 243),
    # ( 81, 172, 219),
    # (217, 167, 204),
# ]

# colors = np.array(colors, dtype='uint8').reshape((1, len(colors), 3))
# colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HLS).reshape(colors.shape[1:])

def color_mask(img):
    # HLS: 0 ~ 180, 0 ~ 255, 0 ~ 255
    table = [
        [(223,  70,  72), ( 20, 120, 100)], # red
        [(244, 166,  57), ( 15,  60, 130)], # orange
        [(250, 219,  77), ( 10,  40,  40)], # yellow
        [(202, 213,  72), ( 15, 120, 120)], # light green
        [(177, 206,  87), ( 15, 120, 120)], # green
        [(151, 215, 243), ( 12, 100, 120)], # light blue
        [( 81, 172, 219), ( 10, 100, 120)], # blue
        [(217, 167, 204), ( 18, 100,  80)], # purple
    ]
    colors = [c[0] for c in table]
    tols = [c[1] for c in table]

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0].astype(int)
    l = hls[:,:,1].astype(int)
    s = hls[:,:,2].astype(int)

    colors = np.array(colors, dtype='uint8')
    colors = colors.reshape((1, len(colors), 3))
    colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HLS).reshape(colors.shape[1:])

    all_mask = np.zeros(h.shape).astype(bool)

    for c, tol in zip(colors, tols):
        c = c.astype(int)
        hmask = np.amin([np.abs(h - c[0]), 180 - np.abs(h - c[0])], axis=0)
        hmask = hmask < tol[0]
        lmask = np.abs(l - c[1]) < tol[1]
        smask = np.abs(s - c[2]) < tol[2]

        all_mask |= hmask & smask & lmask

    return all_mask

    # all_mask = all_mask.astype(int)
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.imshow(img)
    # ax2.imshow(all_mask, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(0)

def get_alpha_pos(k, mask):
    points = []
    for i, row in enumerate(mask):
        for j, v in enumerate(row):
            if v:
                points.append((j, i))
    points = np.array(points)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    return kmeans.cluster_centers_
