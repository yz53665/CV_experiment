'''
实现手动选取并保存模版

@操作：
    按下鼠标左键并拖动选择模板区域, 松开后再次按下可重复选择
    按下鼠标右键确认当前选择的模板, 模板将保存在TamplateCatcher的实例化对象中
    获取模板后可以使用getTemplate/getMask方法获取模板, 或者使用save方法保存模板
@实例：
    from MouseCatchTemplate import TemplateCatcher
    catcher = TemplateCatcher()
    catcher.catchTemplateFrom(src)
    template = catcher.getTemplate()
'''
import copy

import cv2 as cv
import numpy as np


class TemplateCatcher:
    def __init__(self):
        self.__srcImg = None
        self.__tmpImg = None
        self.__prePoint = None
        self.__curPoint = None
        self.__template = None
        self.__mask = None
        self.__templateConfirmed = False

    def catchTemplateFrom(self, srcImg):
        '''
        启动模板捕捉窗口
        '''
        self.__setSrcImgFrom(srcImg)
        self.__testSrcImg()
        self.__catchTemplate()
        self.__showResult()

    def __setSrcImgFrom(self, srcImg):
        self.__srcImg = srcImg

    def __testSrcImg(self):
        if self.__unableToReadSrcImg():
            print("error opening image!")

    def __unableToReadSrcImg(self):
        return self.__srcImg.size == 0

    def __catchTemplate(self):
        cv.namedWindow("catch template", cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback("catch template", self.__mouseControl)
        cv.imshow("catch template", self.__srcImg)
        while not self.__templateConfirmed:
            cv.waitKey(30)


    def __mouseControl(self, event, x, y, flags, param):
        self.__inalizeAllParameters()
        if event is cv.EVENT_LBUTTONDOWN:
            self.__drawCircle(x, y)
        if event is cv.EVENT_RBUTTONUP:
            self.__confirmTemplate()
        if event is cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            self.__drawRectangle(x, y)
        if event is cv.EVENT_LBUTTONUP:
            self.__finishedRectangle(x, y)

    def __inalizeAllParameters(self):
        self.__tmpImg = copy.copy(self.__srcImg)
        self.__template = copy.copy(self.__srcImg)
        self.__mask = np.zeros(self.__srcImg.shape[0:2])

    def __drawCircle(self, x, y):
        self.__prePoint = (int(x), int(y))
        self.__tmpImg = cv.circle(self.__tmpImg, self.__prePoint, 2, (255, 0, 0),
                                cv.FILLED, cv.LINE_AA, 0)
        cv.imshow('catch template', self.__tmpImg)


    def __confirmTemplate(self):
            x1, y1 = self.__prePoint
            x2, y2 = self.__curPoint
            self.__template = self.__template[y1:y2, x1:x2]
            cv.rectangle(self.__mask, self.__prePoint, self.__curPoint, (255, 255, 255),
                         cv.FILLED, cv.LINE_AA)
            self.__templateConfirmed = True

    def __drawRectangle(self, x, y):
            self.__curPoint = (int(x),int(y))
            self.__tmpImg = cv.rectangle(self.__tmpImg, self.__prePoint, self.__curPoint,
                                  (0, 255, 0))
            cv.imshow('catch template', self.__tmpImg)

    def __finishedRectangle(self, x, y):
            # 鼠标左键抬起，画出图像
            self.__curPoint = (int(x),int(y))
            self.__tmpImg =  cv.circle(self.__tmpImg, self.__prePoint, 2, (255, 0, 0),
                                     cv.FILLED, cv.LINE_AA, 0)
            self.__tmpImg = cv.rectangle(self.__tmpImg, self.__prePoint, self.__curPoint,
                                  (0, 255, 0))
            cv.imshow('catch template', self.__tmpImg)

    def __showResult(self):
        cv.imshow("template", self.__template)
        cv.waitKey(0)
        cv.imshow("mask", self.__mask)
        cv.waitKey(0)

    def save(self, dateType=np.uint8):
        '''
        保存template 和 mask 到当前文件夹下, 默认保存的数据格式为np.uint8
        '''
        cv.imwrite("template.png", self.__template.astype(dateType))
        cv.imwrite("mask.png", self.__mask.astype(dateType))

    def getTemplate(self, dateType=np.uint8):
        '''
        默认数据格式 np.uint8
        '''
        return self.__template.astype(dateType)

    def getMask(self, dateType=np.uint8):
        '''
        默认数据格式 np.uint8
        '''
        return self.__template.astype(dateType)


if __name__ == "__main__":
    src = cv.imread('ExpPic/car/473.bmp')
    catcher = TemplateCatcher()
    catcher.catchTemplateFrom(src)
