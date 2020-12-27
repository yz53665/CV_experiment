import copy
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

# 对上一题提取的二值图像进行进一步分割处理
img = cv.imread('binarize.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

dis, cnts, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
imgW, imgH = img.shape[::-1]
divResults = []

for i in cnts:
    x, y, w, h = cv.boundingRect(i)
    area = cv.contourArea(i)
    if area < 300:
        continue
    divResult = img[y:y + h, x:x + w]
    cv.imshow('div', divResult)
    divResults.append(divResult)
    cv.waitKey(0)

cv.imwrite('divA.png', divResults[1])
cv.imwrite('divB.png', divResults[0])

