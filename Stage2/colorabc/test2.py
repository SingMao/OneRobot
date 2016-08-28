import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# np.set_printoptions(threshold=np.inf)

def edge_detect(img):
    BLUR_SIZE = 51
    TRUNC_RATIO = 0.75
    CLOSING_SIZE = 5

    # denoised = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # too_bright=np.logical_and(img[:,:,1]<50, img[:,:,2]>200)
    # np.set_printoptions(threshold=np.nan)
    # np.savetxt('conconcon',img[:,:,1],'%i')
    # img[:,:,1]=np.where(too_bright, np.sqrt(img[:,:,1])+70, img[:,:,1])
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (BLUR_SIZE, BLUR_SIZE))

    edge = np.floor(0.5 * gray + 0.5 * (255 - blur)).astype('uint8')

    hist,bins = np.histogram(edge.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equ = cdf[edge]

    hist,bins = np.histogram(equ.flatten(),256,[0,256])
    max_idx = np.argmax(hist);
    hist_clean = np.where(equ > TRUNC_RATIO * max_idx, 255, equ)

    kernel = np.ones((CLOSING_SIZE, CLOSING_SIZE), np.uint8)
    closing = cv2.morphologyEx(hist_clean, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing, cmap='Greys_r')
    plt.show()
    cv2.waitKey(100)

def color_mask(img):
    # HSV: 0 ~ 180, 0 ~ 255, 0 ~ 255
    table = [
        [(  0, 65, 87), (  8, 40, 50)], # red
        [( 35, 74, 96), ( 12, 20, 50)], # orange
        [( 50, 70, 96), ( 12, 40, 50)], # yellow
        [( 65, 65, 85), ( 12, 40, 50)], # light green
        [( 75, 54, 81), ( 12, 10, 50)], # green
        [(200, 35, 96), ( 12, 10, 50)], # light blue
        [(200, 65, 87), ( 12, 20, 50)], # blue
        [(310, 23, 84), ( 12,  5, 50)], # purple
    ]
    colors = [(c[0] / 2, c[1] * 2.55, c[2] * 2.55) for c, t in table]
    tols = [(t[0] / 2, t[1] * 2.55, t[2] * 2.55) for c, t in table]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0].astype(int)
    s = hsv[:,:,1].astype(int)
    v = hsv[:,:,2].astype(int)
    img = img[:,:,::-1]

    colors = np.array(colors, dtype='uint8')
    # print(colors)

    all_mask = np.zeros(h.shape).astype(bool)

    for c, tol in zip(colors, tols):
        c = c.astype(int)
        hmask = np.amin([np.abs(h - c[0]), 180 - np.abs(h - c[0])], axis=0)
        hmask = hmask < tol[0]
        smask = c[1] - s < tol[1]
        vmask = np.abs(v - c[2]) < tol[2]

        all_mask |= hmask & smask & vmask

    return all_mask

    all_mask = all_mask.astype(int)

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


# img = cv2.imread('Alphabet-Puzzle.jpg')
# img = cv2.imread('puzzle_alphaless.jpg')
# img = cv2.imread('puzzle_headless.jpg')
# img = cv2.imread('puzzle_crop.jpg')
# img = cv2.imread('sim1.jpg')
img = cv2.imread('sim2.jpg')
# img = cv2.imread('IMG_20160629_142046.jpg')
# img = cv2.imread('IMG_20160629_142122.jpg')
# img = cv2.imread('IMG_20160629_142046.jpg')
# edge_detect(img)

mask = color_mask(img)
# centers = get_alpha_pos(26, mask)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(mask.astype(int), cmap='Greys_r')
ax2.imshow(img[:,:,::-1])
# ax2.scatter(x=centers[:,0], y=centers[:,1], c='r', s=400)
plt.show()
