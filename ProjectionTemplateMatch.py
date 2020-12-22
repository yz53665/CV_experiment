'''
实现基于边缘投影的特征提取
'''
from MouseCatchTemplate import catchtemplate
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

imgNum = input('请输入检测图片的数量：')
methodNum = input('请输入检测函数编号（0-5）:')
methodNum = int(methodNum)
imgParDir = 'ExpPic/car/'
# imgParDir = 'ExpPic/plane'
imgDirList = []
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


def ProjectFigure(imgs):
    plt.figure(figsize=(6,6))
    col = (len(imgs) + 1) / 2 
    for index, i in enumerate(imgs):
        plt.subplot(int(col), 2, index + 1)
        plt.plot(i)


for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()

src = cv.imread(imgDirList[0])
# template = catchtemplate(src)
template = cv.imread('template.png')

# 获取模版的边缘
grayTemplate = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
cannyTemplate = cv.Canny(grayTemplate, 50, 150)

projectX = np.sum(cannyTemplate, axis=0)
projectY = np.sum(cannyTemplate, axis=1)
ProjectFigure([projectX, projectY])
plt.show()

projectXs = []
projectYs = []

for i in imgDirList:
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cannySrc = cv.Canny(gray, 50, 150)
    projectX = np.sum(cannySrc, axis=0)
    projectXs.append(projectX)
    projectY = np.sum(cannySrc, axis=1)
    projectYs.append(projectY)
ProjectFigure(projectXs)
ProjectFigure(projectYs)
plt.show()

#      res = cv.matchTemplate(dxy, templateDxy, eval(methods[methodNum]))
    #  minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
    #
    #  if methods[methodNum] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    #      topLeft = minLoc
    #  else:
    #      topLeft = maxLoc
    #  w, h = grayTemplate.shape[::-1]
    #  bottomRight = (topLeft[0] + w, topLeft[1] + h)
    #
    #  cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 1)
    #
    #  cv.namedWindow('grey')
    #  cv.imshow('grey', res)
    #  cv.namedWindow(i)
    #  cv.imshow(i, img)
    #  cv.waitKey(0)
