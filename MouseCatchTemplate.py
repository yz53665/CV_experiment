import cv2
import copy
import numpy as np 
from sys import argv

src = 0
prePoint = 0
curPoint = 0
template = 0

def mousehandle(event, x, y, flags, usrDat):
    global src
    global prePoint
    global curPoint
    global template
    img = copy.copy(src)
    tmpImg = copy.copy(src)
    template = copy.copy(src)
    if event == cv2.EVENT_LBUTTONDOWN:
        img = copy.copy(src)
        prePoint = (int(x),int(y))
        img =  cv2.circle(img, prePoint, 2, (255, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
        cv2.imshow('source',img)
    elif event == cv2.EVENT_RBUTTONUP:
        # 如果鼠标右键松开，则确认模版
        x1, y1 = prePoint
        x2, y2 = curPoint
        template = template[y1:y2, x1:x2]
        cv2.imwrite("template.png", template)
        cv2.namedWindow('template', cv2.WINDOW_NORMAL)
        cv2.imshow('template', template)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        # 如果鼠标左键被按下且鼠标在移动, 在图像上画矩形
        tmpImg = copy.copy(src)
        print(prePoint)
        curPoint = (int(x),int(y))
        tmpImg = cv2.rectangle(tmpImg, prePoint, curPoint, (0, 255, 0))
        cv2.imshow('source', tmpImg)
    elif event == cv2.EVENT_LBUTTONUP:
        # 鼠标左键抬起，画出图像
        img = copy.copy(src)
        curPoint = (int(x),int(y))
        img =  cv2.circle(img, prePoint, 2, (255, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
        tmpImg = cv2.rectangle(tmpImg, prePoint, curPoint, (0, 255, 0))
        cv2.imshow('source', tmpImg)

# 实现手动提取模版
def catchtemplate(Image):
    global src
    src = Image
    if src.size == 0:
        print("error opining image" )
        return
    cv2.namedWindow('source', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('source', mousehandle)
    cv2.imshow('source', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return template

if __name__ == '__main__':
    src = cv2.imread('ExpPic/plane/P000.bmp')
    catchtemplate(src)

