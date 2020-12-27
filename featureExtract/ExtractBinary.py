import cv2 as cv 
import matplotlib.pyplot as plt

path = 'ab.jpg'

img = cv.imread(path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 展示直方图
plt.figure()
plt.hist(gray.ravel(), 256, (0,255))

# 大津法阈值分割
thr, binImg = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
print('分割阈值:' + str(thr))

element = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)) #使用3*3大小椭圆型的结构元
dilImg= cv.morphologyEx(binImg, cv.MORPH_DILATE, element) #进行开运算

plt.figure()
plt.imshow(dilImg, cmap='gray')
plt.show()

cv.imwrite('binarize.png', dilImg)
