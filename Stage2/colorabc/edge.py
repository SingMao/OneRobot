import numpy as np
import cv2

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

# np.set_printoptions(threshold=np.inf)
def find_block(img):
    NEW_WIDTH = 600
    NEW_HEIGHT = 600

    # binary = edge_detect(img, mode='block', blur_size=9, erode_size=5, dilate_size=5)
    binary = cv2.Canny(img, 20, 100)
    cv2.imshow('lalala', binary)
    cv2.waitKey(10)
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    real_contour = None
    largest_area = 0
    approx_contour = None
    print('-'*80)
    for c in contours:
        area = cv2.contourArea(c) / (img.shape[0] * img.shape[1])
        if area > 0.1:
            print(area)
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

    cv2.drawContours(img, [real_contour], -1, (255, 0, 0), 5)
    cv2.imshow('qq', img)
    cv2.waitKey(10)

    if real_contour is None or len(approx_contour) != 4:
        # if real_contour is None:
            # print("REEEEEEEEEEEEEEEEL")
            # if approx_contour is not None:
                # print(approx_contour)
        # else:
            # print("444444444444444444")
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

    return pers_src, dst, largest_area
    # plt.imshow(dst[:,:,::-1])
    # plt.show()
    # cv2.waitKey(100)
