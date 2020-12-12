import numpy as np
import cv2

img = cv2.imread('ExpPic/car/473.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 20)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

cv2.namedWindow('corner')
cv2.imshow('corner', img)
cv2.waitKey(0)
