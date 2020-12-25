'''
实现基于Canny算子的边缘轮廓特征的模版匹配
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

def GetCannyMatch(cannyImg, cannyTemplate, method):
    res = cv.matchTemplate(cannyImg, cannyTemplate, method)
    cv.normalize(res, res, 0, 1, cv.NORM_MINMAX, -1)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
    return minVal, maxVal, minLoc, maxLoc

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

for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()
imgDirList = imgDirList[0:int(imgNum)]

src = cv.imread(imgDirList[0])
template, mask = catchtemplate(src)
cannyTemplate = GetCanny(template)

kal = Kalman2D()

for i in imgDirList:
    img = cv.imread(i)
    cannySrc = GetCanny(img)
    minVal, maxVal, minLoc, maxLoc = GetCannyMatch(cannySrc, cannyTemplate, eval(methods[methodNum]))

    if methods[methodNum] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    
    kal.update(topLeft[0], topLeft[1])
    predict = kal.getPredict()
    estimate = kal.getEstimate()
    
    w, h =cannyTemplate.shape[::-1]
    bottomRight = (topLeft[0] + w, topLeft[1] + h)
    predictBottom = (predict[0] + w, predict[1] + h)
    estimateBottom = (estimate[0] + w, estimate[1] + h)
    
    cv.rectangle(img, topLeft, bottomRight, (0, 255, 0), 1) #当前位置标记绿色
    cv.rectangle(img, predict, predictBottom, (0, 0, 255), 1)   #预测下一帧位置标记红色
    cv.rectangle(img, estimate, estimateBottom, (255, 0, 0), 1) #估计当前位置标记蓝色

    cv.namedWindow(i)
    cv.imshow(i, img)
    cv.waitKey(0)

