import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)

# np.set_printoptions(threshold=np.inf)
def edge_detect(img, mode, blur_size, erode_size, dilate_size):

    BLUR_SIZE = blur_size # 81 for large/19 for small 
    TRUNC_RATIO = 0.85
    ERODE_SIZE = erode_size
    DILATE_SIZE = dilate_size

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

    erode_kernel = np.ones((ERODE_SIZE, ERODE_SIZE), np.uint8)
    dilate_kernel = np.ones((DILATE_SIZE, DILATE_SIZE), np.uint8)
    if mode == 'alpha':
        dilation = cv2.dilate(hist_clean, dilate_kernel, iterations = 1)
        erosion = cv2.erode(dilation, erode_kernel, iterations = 1)
        closing = erosion
    else:
        erosion = cv2.erode(hist_clean, erode_kernel, iterations = 1)
        dilation = cv2.dilate(erosion, dilate_kernel, iterations = 1)
        closing = dilation
    # closing = cv2.morphologyEx(hist_clean, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(closing, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(100)
    binary = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('洪士號', binary)
    return binary

    # plt.imshow(binary, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(100)
    # exit()

def find_alphabet_contours(img):
    AREA_MIN_THRES = 0.005
    AREA_MAX_THRES = 0.15

    binary = edge_detect(img, mode='alpha', blur_size=19, erode_size=5, dilate_size=3)
    # plt.imshow(binary, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(100)

    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    real_contours = []
    # return contours
    for c in contours:
        area = cv2.contourArea(c) / (img.shape[0] * img.shape[1])
        if area > AREA_MIN_THRES and area < AREA_MAX_THRES:
            real_contours.append(c)
    return real_contours

def find_block(img):
    NEW_WIDTH = 600
    NEW_HEIGHT = 600

    binary = edge_detect(img, mode='block', blur_size=9, erode_size=5, dilate_size=5)
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    real_contour = None
    largest_area = 0
    approx_contour = None
    for c in contours:
        area = cv2.contourArea(c) / (img.shape[0] * img.shape[1])
        # print(area)
        # if area < 0.0005 or area > 0.3:
        if area < 0.25 or area > 0.85:
            continue
        if real_contour is None or area > largest_area:
            real_contour = c
            largest_area = area
            epsilon = 0.1 * cv2.arcLength(real_contour, True)
            approx_contour = cv2.approxPolyDP(real_contour, epsilon, True)
            # if len(approx_contour) != 4:
                # real_contour = None

    if real_contour is None or len(approx_contour) != 4:
        if real_contour is None:
            print("REEEEEEEEEEEEEEEEL")
            if approx_contour is not None:
                print(approx_contour)
        else:
            print("444444444444444444")
        raise ValueError('A very specific bad thing happened')

    # plt.imshow(binary, cmap='Greys_r')
    # plt.show()
    # cv2.waitKey(100)

    # cv2.drawContours(img, [real_contour], -1, (255, 0, 0), 5)
    # plt.imshow(img[:,:,::-1])
    # plt.show()
    # cv2.waitKey(100)

    pers_src = np.float32([x[0] for x in approx_contour][:4])
    # print('Edges :', len(approx_contour))
    idx = np.argmin(pers_src[:,0] + pers_src[:,1])
    perm = list(range(idx, 4)) + list(range(0, idx))
    pers_src = pers_src[perm, :]
    pers_dst = np.float32([(0, 0), (NEW_WIDTH, 0), (NEW_WIDTH, NEW_HEIGHT), (0, NEW_HEIGHT)])

    pers_transform = cv2.getPerspectiveTransform(pers_src, pers_dst)

    dst = cv2.warpPerspective(img, pers_transform, (NEW_WIDTH, NEW_HEIGHT))

    return pers_src, dst
    # plt.imshow(dst[:,:,::-1])
    # plt.show()
    # cv2.waitKey(100)



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
    # print(colors)
    # return
    
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


while True:
    ok, img = cap.read()
    # print(img.shape if ok else 'Q__Q')
    if not ok:
        continue

    try:
        corners, dst = find_block(img)
    except ValueError:
        print('Q__Q')
    else:
        # mask = (255 * color_mask(dst)).astype('uint8')
        # _, maxval, _, maxpos = cv2.minMaxLoc(img[:,:,2])
        # print(maxval, maxpos)
        # if maxval >= 150:
            # cv2.circle(img, maxpos, 20, (0, 255, 0), 5)
        # cv2.circle(img, (320, 240), 10, (255, 0, 0), 6)
        # cv2.circle(img, (400, 320), 10, (255, 0, 0), 10)

        print('got')
        cv2.fillPoly(img, pts=[corners.astype('int32')], color=(255, 255, 255))
        # mask = color_mask(img).astype('uint8') * 255
        alpha_contours = find_alphabet_contours(img)
        for c in alpha_contours:
            center = np.reshape(np.mean(c, 0), 2).astype(int)
            cv2.circle(img, (center[0], center[1]), 10, (255, 0, 0), 6)
        # cv2.drawContours(img, alpha_contours, -1, (255, 0, 0), 5)

        # cv2.imshow('test', mask)
        cv2.imshow('test', img)

    cv2.waitKey(10)

