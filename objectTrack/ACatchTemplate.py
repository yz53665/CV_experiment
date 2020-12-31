'''
实现自动检测汽车并保存模版
'''
import numpy as np
import cv2
import os

img = cv2.imread('ExpPic/car/473.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgParDir = 'testcar/'
imgDirList = []

# 批量读取某一文件夹下的所有文件
for info in os.listdir(imgParDir):
    imgDirList.append(os.path.join(imgParDir, info))
imgDirList.sort()
print(imgDirList)

carCascade = cv2.CascadeClassifier('cars.xml')

# cars = carCascade.detectMultiScale(gray, 1.1, 3)
#  for index, (x, y, w, h) in enumerate(cars):
    #  img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    #  template = img[y:y+h, x:x+w]
    #  cv2.imwrite('template' + str(index)+ '.png', template)
#      cv2.imshow('template', template)
# cv2.imwrite('autoTemplate.png', img)
for index, i in enumerate(imgDirList):
    img = cv2.imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = carCascade.detectMultiScale(gray, 1.1, 3)
    for index, (x, y, w, h) in enumerate(cars):
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite('test/car' + str(index)+ '.png', img)

    cv2.imshow('img', img)
    cv2.waitKey(0)
