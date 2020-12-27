import copy
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

imgA = cv.imread('divA.png', cv.IMREAD_GRAYSCALE)
imgB = cv.imread('divB.png', cv.IMREAD_GRAYSCALE)

# 求解七个不变矩
momentsA = cv.moments(imgA, True)
momentsB = cv.moments(imgB, True)
huA = cv.HuMoments(momentsA)
huB = cv.HuMoments(momentsB)

for index, i in enumerate(zip(huA, huB)):
    a, b = i
    print('Hu{} for A and B: {} {}'.format(index + 1, a, b))
