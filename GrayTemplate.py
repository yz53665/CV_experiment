'''
手动选择模版，实现基于灰度的模版匹配方法
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
imgDirList = imgDirList[0:int(imgNum)]

src = cv.imread(imgDirList[0])
template, mask = catchtemplate(src)
# template = cv.imread('template.png')
grayTemplate = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
# 提取模版宽和高
if len(template.shape) == 3:
    channels, w, h = template.shape[::-1]
else:
    w, h = template[::-1]

for i in imgDirList:
    # 对每一张图片进行全局模版匹配
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    res = cv.matchTemplate(gray, grayTemplate, eval(methods[methodNum]))
    cv.normalize(res, res, 0, 1, cv.NORM_MINMAX, -1)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)

    if methods[methodNum] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + w, topLeft[1] + h)

    cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 1)
   
    cv.namedWindow('grey')
    cv.imshow('grey', res)
    cv.waitKey(500)
    cv.namedWindow(i)
    cv.imshow(i, img)
    cv.waitKey(200)


