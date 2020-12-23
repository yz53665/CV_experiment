'''
实现基于卡尔曼滤波
'''
from MouseCatchTemplate import catchtemplate
import numpy as np
import cv2 as cv
import os

imgNum = input('请输入检测图片的数量：')
methodNum = input('请输入检测函数编号（0-5）:')
methodNum = int(methodNum)
imgParDir = 'ExpPic/car/'
#imgParDir = 'ExpPic/plane'
imgDirList = []
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# 批量读取某一文件夹下的所有文件
for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()

src = cv.imread(imgDirList[0])
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
template, mask = catchtemplate(src)
#template = cv.imread('template.png')

points = cv.goodFeaturesToTrack(gray, 5, 0.1, 10, mask=mask)
for i in points:
    x, y = i.ravel()
    cv.circle(src, (x, y), 2, (255, 0, 0))
cv.imshow('src', src)
cv.waitKey(0)

for i in imgDirList:
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    points = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)

    #  res = cv.matchTemplate(img, template, eval(methods[methodnum]))
    #  minval, maxval, minloc, maxloc = cv.minmaxloc(res)
    #
    #  if methods[methodnum] in [cv.tm_sqdiff, cv.tm_sqdiff_normed]:
    #      topleft = minloc
    #  else:
    #      topleft = maxloc
    #  bottomright = (topleft[0] + w, topleft[1] + h)
    #
    #  cv.rectangle(img, topleft, bottomright, (0, 255, 0), 1)
 #
    #  cv.namedwindow('grey')
    #  cv.imshow('grey', res)
    #  cv.namedwindow(i)
    #  cv.imshow(i, img)
    #  cv.waitkey(0)
    #
   #
