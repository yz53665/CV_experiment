'''
实现手动选取并保存模版
'''
import copy
from sys import argv

import cv2 as cv
import numpy as np


class TemplateCatcher:
    def __init__(self):
        self.srcImg = None
        self.tmpImg = None
        self.prePoint = None
        self.curPoint = None
        self.template = None
        self.mask = None
        self.x = None
        self.y = None
        self.event = None
        self.flags = None
        self.mouseResponse = [self.__drawCircle, self.__confirmTemplate,
                              self.__drawRectangle,
                              self.__finishedRectangle]

    def catchTemplateFrom(self, srcImg):
        self.__setSrcImgFrom(srcImg)
        self.__catchTemplate()
        self.__showResult()
        self.__saveResult()

    def __setSrcImgFrom(self, srcImg):
        self.srcImg = srcImg
        self.testSrcImg()

    def testSrcImg(self):
        if self.__unableToReadSrcImg():
            print("error opening image!")

    def __unableToReadSrcImg(self):
        return self.srcImg.size == 0

    def __catchTemplate(self):
        cv.namedWindow("catch template", cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback("catch template", self.mouseHandle)

    def mouseHandle(self, event, x, y, flags, usrDat):
        self.__inalizeParameters(event, flags, self.x, self.y)
        for fun in self.mouseResponse:
            fun()

    def __inalizeAllParameters(self, event, flags, x, y):
        self.tmpImg = copy.copy(self.srcImg)
        self.template = copy.copy(self.srcImg)
        self.mask = np.zeros(self.srcImg.shape[0:2])
        self.event = event
        self.flags = flags

    def __showResult(self):
        cv.imshow("srcImg", self.srcImg)
        cv.waitKey(0)
        cv.imshow("template", self.template)
        cv.waitKey(0)
        cv.imshow("mask", self.mask)
        cv.waitKey(0)

    def __saveResult(self):
        cv.imwrite("template.png", self.template)
        cv.imwrite("mask.png", self.mask)

    def __drawCircle(self):
        if self.event is cv.EVENT_LBUTTONDOWN:
            self.prePoint = (int(self.x), int(self.y))
            self.tmpImg = cv.circle(self.tmpImg, self.prePoint, 2, (255, 0, 0),
                                    cv.FILLED, cv.LINE_AA, 0)
            cv.imshow('source', self.tmpImg)

    def __confirmTemplate(self, event):
        if self.event is cv.EVENT_RBUTTONUP:
            x1, y1 = self.prePoint
            x2, y2 = self.curPoint
            template = template[y1:y2, x1:x2]
            cv.rectangle(mask, self.prePoint, self.curPoint, (255, 255, 255),
                         cv.FILLED, cv.LINE_AA)

    def __drawRectangle(self):
        if self.event is cv.EVENT_MOUSEMOVE and self.flags == cv.EVENT_FLAG_LBUTTON:
            self.curPoint = (int(self.x),int(self.y))
            self.tmpImg = cv.rectangle(self.tmpImg, self.prePoint, self.curPoint,
                                  (0, 255, 0))
            cv.imshow('source', self.tmpImg)

    def __finishedRectangle(self):
        if self.event is cv.EVENT_LBUTTONUP:
            # 鼠标左键抬起，画出图像
            self.curPoint = (int(self.x),int(self.y))
            self.tmpImg =  cv.circle(self.tmpImg, self.prePoint, 2, (255, 0, 0),
                                     cv.FILLED, cv.LINE_AA, 0)
            self.tmpImg = cv.rectangle(self.tmpImg, self.prePoint, self.curPoint,
                                  (0, 255, 0))
            cv.imshow('source', self.tmpImg)


def mousehandle(event, x, y, flags, usrDat):
    img = copy.copy(src)
    tmpImg = copy.copy(src)
    template = copy.copy(src)
    mask = np.zeros(src.shape[0:2])
    if event is cv.EVENT_LBUTTONDOWN:
        img = copy.copy(src)
        prePoint = (int(x),int(y))
        img =  cv.circle(img, prePoint, 2, (255, 0, 0), cv.FILLED, cv.LINE_AA, 0)
        cv.imshow('source',img)
    elif event is cv.EVENT_RBUTTONUP:
        # 如果鼠标右键松开，则确认模版
        x1, y1 = prePoint
        x2, y2 = curPoint
        template = template[y1:y2, x1:x2]
        cv.rectangle(mask, prePoint, curPoint, (255, 255, 255), cv.FILLED,
                     cv.LINE_AA)
    elif event is cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        # 如果鼠标左键被按下且鼠标在移动, 在图像上画矩形
        tmpImg = copy.copy(src)
        curPoint = (int(x),int(y))
        tmpImg = cv.rectangle(tmpImg, prePoint, curPoint, (0, 255, 0))
        cv.imshow('source', tmpImg)
    elif event is cv.EVENT_LBUTTONUP:
        # 鼠标左键抬起，画出图像
        img = copy.copy(src)
        curPoint = (int(x),int(y))
        img =  cv.circle(img, prePoint, 2, (255, 0, 0), cv.FILLED, cv.LINE_AA, 0)
        tmpImg = cv.rectangle(tmpImg, prePoint, curPoint, (0, 255, 0))
        cv.imshow('source', tmpImg)

# 实现手动提取模版
def catchTemplate(Image):
    global src
    global mask
    global template
    src = Image
    if src.size == 0:
        print("error opining image" )
        return
    cv.namedWindow('source', cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback('source', mousehandle)
    cv.imshow('source', src)
    cv.waitKey(0)
    cv.imshow('mask', mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    mask = mask.astype(np.uint8)
    return template, mask


src = cv.imread('ExpPic/car/473.bmp')
catcher = TemplateCatcher()
catcher.catchTemplateFrom(src)
