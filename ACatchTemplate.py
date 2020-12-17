'''
实现自动检测汽车并保存模版
'''
import numpy as np
import cv2

img = cv2.imread('ExpPic/car/473.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

carCascade = cv2.CascadeClassifier('/Users/qiuruiqi/qrqCode/opencv-car-detection/cars.xml')

cars = carCascade.detectMultiScale(gray, 1.1, 3)
for index, (x, y, w, h) in enumerate(cars):
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    template = img[y:y+h, x:x+w]
    cv2.imwrite('template' + str(index)+ '.png', template)
    cv2.imshow('template', template)

cv2.imshow('img', img)
cv2.waitKey(0)
