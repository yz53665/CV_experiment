'''
实现基于卡尔曼滤波的边缘轮廓特征的物体预测与模版匹配
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
#imgParDir = 'ExpPic/plane'
imgDirList = []
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

def GetCanny(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(img, 50, 150)
    return canny

def CannyMatch(cannyImg, cannyTemplate, method):
    res = cv.matchTemplate(cannyImg, cannyTemplate, method)
    cv.normalize(res, res, 0, 1, cv.NORM_MINMAX, -1)
    return res

class Kalman2D(object):

    def __init__(self, processNoise=1e-5, measurementNoise=1e-1, error=0.1):
        self.kalman = cv.KalmanFilter(4, 2, 0)
        self.kalman.transitionMatrix = np.array([[1.,0.,1.,0.],
                                                [0.,1.,0.,1.],
                                                [0.,0.,1.,0.],
                                                [0.,0.,0.,1.]])
        self.kalman.measurementMatrix = np.array([[1., 0., 0., 0.],
                                                 [0., 1., 0., 0.]])
        self.kalman.processNoiseCov = processNoise * np.eye(4)
        self.kalman.measurementNoiseCov = measurementNoise * np.eye(2)
        self.kalman.errorCovPost = error * np.ones((4,4))
        self.kalman.statePost = 0.1 * np.random.randn(4,1)
        
        self.predicted = None
        self.estimated = None
        self.kalman_measurement = np.ones((2,1))
    
    def update(self, x, y):
        self.kalman_measurement[0, 0] = x
        self.kalman_measurement[1, 0] = y

        self.predicted = self.kalman.predict()
        self.corrected = self.kalman.correct(self.kalman_measurement)

    def getEstimate(self):
        return int(self.corrected[0,0]), int(self.corrected[1,0])

    def getPredict(self):
        return int(self.predicted[0,0]), int(self.corrected[1,0])


def GetSmallerSrc(src, prePoint, w, h):
    newSrc = src[prePoint[1]:prePoint[1] + 2 * h, prePoint[0]:prePoint[0] + 2 * w]
    return newSrc


def PartialMatch(src, template, pEstimate, method):
    w, h = template.shape[::-1]
    prePoint = (pEstimate[0] - 0.2 * w, pEstimate[1] - 0.2*h)
    prePoint = np.asarray(prePoint, dtype=np.int16)
    if any(prePoint < 0):   # 判断是否超出边界
        res = CannyMatch(src, template, method)
    else:
        newSrc = GetSmallerSrc(src, prePoint, w, h)
        res = CannyMatch(newSrc, template, method)
    return res, prePoint


def PartialTemplateMatch(src, template, pEstimate, method, turn):
    if turn > 3:
        res, prePoint = PartialMatch(src, template, pEstimate, method)
    else:
        res = CannyMatch(src, template, method)    
        prePoint = [0,0]
    prePoint = np.asarray(prePoint)
    return res, prePoint


for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()
imgDirList = imgDirList[0:int(imgNum)]

src = cv.imread(imgDirList[0])
template, mask = catchtemplate(src)
cannyTemplate = GetCanny(template)

kal = Kalman2D(processNoise=7e-2, measurementNoise=1e-2,error=0.5)
kal.update(0, 0)

for index, i in enumerate(imgDirList):
    img = cv.imread(i)
    cannySrc = GetCanny(img)
    w, h = cannyTemplate.shape[::-1]

    predict = kal.getPredict()
    estimate = kal.getEstimate()

    # 进行对轨迹的预测与局部模版提取
    res, prePoint = PartialTemplateMatch(cannySrc, cannyTemplate, estimate ,eval(methods[methodNum]), index)
    curPoint = (prePoint[0] + 2*w, prePoint[1] + 2*h)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
    # 全局模版匹配
    res2 = CannyMatch(cannySrc, cannyTemplate, eval(methods[methodNum]))
    minVal2, maxVal2, minLoc2, maxLoc2 = cv.minMaxLoc(res2)

    if methods[methodNum] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        topLeft = minLoc
        topLeft2 = minLoc2
    else:
        topLeft = maxLoc
        topLeft2 = maxLoc2

    topLeft = topLeft + prePoint
    
    bottomRight = (topLeft[0] + w, topLeft[1] + h)
    bottomRight2 = (topLeft2[0] + w, topLeft2[1] + h)
    topLeft = (topLeft[0], topLeft[1])
    prePoint = (prePoint[0], prePoint[1])
    
    cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 1) #当前位置标记绿色
    cv.rectangle(img, topLeft2, bottomRight2, (255, 0, 0), 1) #全局模版匹配标记绿色
    if index > 2:
        cv.rectangle(img, prePoint, curPoint, (0, 0, 255), 1)   #画出用于局部匹配模版的区域

    kal.update(topLeft[0], topLeft[1])

    cv.imshow(i,img)
    cv.waitKey(0)

