import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

imgParDir = 'car/'
svaeFileAs = 'detectCar'
saveBinary = 'detectBinary'
imgDirList = []
for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))

for index, i in enumerate(imgDirList):
    src = cv.imread(i)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(hsv)
    lowBound0 = np.array([155, 43, 35])
    upBound0 = np.array([180, 255, 255])
    mask0 = cv.inRange(hsv, lowBound0, upBound0)
    lowBound1 = np.array([0, 43, 35])
    upBound1 = np.array([11, 255, 255])
    mask1 = cv.inRange(hsv, lowBound1, upBound1)
    redObjectImg = mask0 + mask1

    elementRllipse = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)) #使用 3*3 大小椭圆型的结构元
    elementCross = cv.getStructuringElement(cv.MORPH_CROSS,(9,5)) #使用 9*5 大小十字型的结构元
    redObjectImg = cv.morphologyEx(redObjectImg, cv.MORPH_OPEN, elementRllipse) #进行开运算
    redObjectImg = cv.morphologyEx(redObjectImg, cv.MORPH_DILATE, elementCross) #进行 2 次膨胀运算
    redObjectImg = cv.morphologyEx(redObjectImg, cv.MORPH_DILATE, elementCross) 

    dis, cnts, hier = cv.findContours(redObjectImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    areas = []
    for i in cnts:  # 计算所有轮廓的面积
        area = cv.contourArea(i)
        areas.append(area)
    indexMax = areas.index(max(areas))  # 得到面积最大轮廓的下标

    x, y, w, h = cv.boundingRect(cnts[indexMax])
    cv.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('car', redObjectImg)
    cv.waitKey(0)
    cv.imshow('car', src)
    cv.waitKey(0)
    cv.imwrite(svaeFileAs + str(index) + '.png', src)
    cv.imwrite(saveBinary + str(index) + '.png', redObjectImg)
