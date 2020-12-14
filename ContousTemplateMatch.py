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

for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()

src = cv.imread(imgDirList[0])
template = catchtemplate(src)

# 获取模版的边缘
grayTemplate = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dx = cv.Sobel(grayTemplate, cv.CV_32F, 1, 0, 1)
dy = cv.Sobel(grayTemplate, cv.CV_32F, 0, 1, 1)
dxy = np.sqrt(dx**2 + dy**2)
cv.namedWindow('dx')
cv.namedWindow('dy')
cv.namedWindow('dxy')
cv.imshow('dx', dx)
cv.imshow('dy', dy)
cv.imshow('dxy', dxy)
cv.waitKey(0)

if len(template.shape) == 3:
    channels, w, h = template.shape[::-1]
else:
    w, h = template[::-1]

for i in imgDirList:
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dx = cv.Sobel(gray, cv.CV_32F, 1, 0)
    dy = cv.Sobel(gray, cv.CV_32F, 0, 1)
