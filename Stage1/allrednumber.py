import cv2
import numpy as np

img = cv2.imread('redgreen/green4.jpg', cv2.CV_LOAD_IMAGE_UNCHANGED)
# img = cv2.resize(img, (320, 240))
output = img.copy()

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# lower_red = np.array([0,100,100], dtype=np.uint8)
# upper_red = np.array([10,255,255], dtype=np.uint8)

# mask = cv2.inRange(hsv, lower_red, upper_red)
# res = cv2.bitwise_and(img, img, mask= mask)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200, 3)

contours, hie = cv2.findContours(canny, cv2.cv.CV_RETR_LIST, cv2.cv.CV_CHAIN_APPROX_SIMPLE)


good_contour = None
good_cen, good_r = None, None
goodness = 0

for con in contours:
    area = cv2.contourArea(con)
    if area < 10: continue
    cen, r = cv2.minEnclosingCircle(con)
    cen_area = np.pi * r * r
    rat = area / cen_area
    if rat > goodness and rat > 0.7:
        goodness = rat
        good_contour = con
        good_cen, good_r = cen, r

print(good_cen, good_r, goodness)

mask = np.zeros(img.shape).astype(np.float32)
cv2.drawContours(mask, [good_contour], -1, 1, cv2.cv.CV_FILLED)
mask = mask[:,:,0]
imgcir = (img * mask[:,:,np.newaxis]).astype(np.uint8)

avg_color = np.sum(imgcir, (0, 1)) / np.sum(mask)

cv2.drawContours(img, [good_contour], -1, avg_color/2, 5)
cv2.imshow('img', img)
cv2.waitKey(0)



exit(0)

# blah = cv2.Canny(img, 100, 200).astype(np.float32)
# while True:
    # cv2.imshow('hh', gray)
    # cv2.waitKey(100)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 5)
print(circles)
 
# ensure at least some circles were found
if circles is not None:
    print('verygoodnumber')
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
 
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
    # show the output image
    while True:
        cv2.imshow("output", np.hstack([img, output]))
        cv2.waitKey(100)
else:
    print('chuanweihao')
