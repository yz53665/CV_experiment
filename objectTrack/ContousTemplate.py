'''
实现基于Canny算子的边缘轮廓特征的模版匹配
'''
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from MouseCatchTemplate import TemplateCatcher

imgNum = input('请输入检测图片的数量：')
methodNum = input('请输入检测函数编号（0-5）:')
methodNum = int(methodNum)
imgParDir = 'ExpPic/car/'
#imgParDir = 'ExpPic/plane'
imgDirList = []
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()


catcher = TemplateCatcher()
catcher.catchTemplateFrom(src)
template = catcher.getTemplate()

grayTemplate = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
templateBinary = cv.Canny(grayTemplate, 50, 150)
cv.imwrite('canny/binTemplate.png', templateBinary)
cv.imshow('template', templateBinary)
cv.waitKey(0)

for index, i in enumerate( imgDirList):
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    srcBinary = cv.Canny(gray, 50, 150)
    cv.imshow('src', srcBinary)
    cv.imwrite('canny/bin' + str(index) + '.png', srcBinary)
    cv.waitKey(0)

    res = cv.matchTemplate(srcBinary, templateBinary, eval(methods[methodNum]))
    cv.normalize(res, res, 0, 1, cv.NORM_MINMAX, -1)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)

    if methods[methodNum] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc

    w, h = grayTemplate.shape[::-1]
    bottomRight = (topLeft[0] + w, topLeft[1] + h)

    cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 1)

    cv.namedWindow('grey')
    cv.imshow('grey', res)
    cv.namedWindow(i)
    cv.imshow(i, img)
    cv.imwrite('canny/img' + str(index) + '.png', img)
    cv.waitKey(0)

