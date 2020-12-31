import copy
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

def GetRoundness(area, length):
    return 4 * np.pi * area / (length ** 2)
def GetComplexity(area, length):
    return length ** 2 / area

imgA = cv.imread('divA.png', cv.IMREAD_GRAYSCALE)
imgB = cv.imread('divB.png', cv.IMREAD_GRAYSCALE)

disA, cntsA, hierA = cv.findContours(imgA, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
disB, cntsB, hierB = cv.findContours(imgB, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

# 计算孔的数量
def CountHole(hier, idFirstChild):
    if idFirstChild == -1:
        return 0
    nHole = 0
    hole = hier[idFirstChild]
    nHole += 1
    while hole[0] != -1:
        hole = hier[hole[0]]
        nHole += 1
    return nHole

# 计算欧拉数,参数需要使用CCOMP方法作为参数的findControus来得到
def GetEuler(hier):
    nOutline = 0
    nHole = 0
    hier = hier[0]
    for i in hier:
        if i[2] == -1:
            continue
        nOutline += 1
        idFirstChild = i[2]
        nHole += CountHole(hier, idFirstChild)
    return nOutline - nHole

eulerA = GetEuler(hierA)
eulerB = GetEuler(hierB)

print('A的欧拉数: {}'.format(eulerA))
print('B的欧拉数: {}'.format(eulerB))

disA, cntsA, hierA = cv.findContours(imgA, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
disB, cntsB, hierB = cv.findContours(imgB, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# 计算面积和周长
areaA = cv.contourArea(cntsA[0])
areaB = cv.contourArea(cntsB[0])
lengthA = cv.arcLength(cntsA[0], True)
lengthB = cv.arcLength(cntsB[0], True)
print('A的周长和面积分别为: {:.2f} {}'.format(lengthA, areaA))
print('B的周长和面积分别为: {:.2f} {}'.format(lengthB, areaB))

# 计算圆形度
RA = GetRoundness(areaA, lengthA)
RB = GetRoundness(areaB, lengthB)
print('A的圆形度: {:.2f}'.format(RA))
print('B的圆形度: {:.2f}'.format(RB))

# 计算形状复杂性
EA = GetComplexity(areaA, lengthA)
EB = GetComplexity(areaB, lengthB)
print('A的形状复杂性: {:.2f}'.format(EA))
print('B的形状复杂性: {:.2f}'.format(EB))



