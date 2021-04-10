'''
实现基于sobel算子的边缘特征的模版匹配
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

def SobelNormalize(img):
    dx = cv.Sobel(img, cv.CV_32F, 1, 0, 1)
    plt.hist(dx.ravel(), 256, (0,255))
    plt.show()
    dy = cv.Sobel(img, cv.CV_32F, 0, 1, 1)
    plt.hist(dy.ravel(), 256, (0,255))
    plt.show()
    dx = cv.convertScaleAbs(dx)
    plt.hist(dx.ravel(), 256, (0,255))
    plt.show()
    dy = cv.convertScaleAbs(dy)
    plt.hist(dy.ravel(), 256, (0,255))
    plt.show()
    dxy = cv.addWeighted(dx, 0.5, dy, 0.5, 0, dtype=cv.CV_32F)
    return dxy

for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()

src = cv.imread(imgDirList[0])
catcher = TemplateCatcher()
catcher.catchTemplateFrom(src)
template = catcher.getTemplate()

sobelTemplate = SobelNormalize(template)

# 获取模版的边缘
grayTemplate = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
templateDxy = SobelNormalize(grayTemplate)

for i in imgDirList:
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dxy = SobelNormalize(gray)
    
    # res = cv.matchTemplate(dxy, templateDxy, eval(methods[methodNum]))
    for index, i in enumerate(methods):
        res = cv.matchTemplate(dxy, templateDxy, eval(i))
        cv.normalize(res, res, 0, 1, cv.NORM_MINMAX, -1)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)

        if methods[methodNum] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            topLeft = minLoc
        else:
            topLeft = maxLoc
        w, h = grayTemplate.shape[::-1]
        bottomRight = (topLeft[0] + w, topLeft[1] + h)

        cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 1)
        cv.putText(img, str(index), topLeft, cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
   
    cv.namedWindow('grey')
    cv.imshow('grey', res)
    cv.namedWindow(i)
    cv.imshow(i, img)
    cv.waitKey(0)

