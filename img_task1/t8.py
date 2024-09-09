import cv2
import numpy as np

img = cv2.imread("8.tif",-1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# 定义橙色的HSV范围
# 这里假设橙色的范围为低阈值和高阈值
lower_orange = np.array([11, 43, 46])
upper_orange = np.array([20, 255, 255])


# 根据阈值构建掩码
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# 使用形态学操作使颜色区域更加完整
# 创建形态学操作的结构元素
kernel = np.ones((10, 10), np.uint8)

# 进行闭运算（先膨胀后腐蚀）以填充小孔
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 进行开运算（先腐蚀后膨胀）以去除噪点
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=2)

cv2.imshow('mask', mask)
# 通过查找轮廓来提取橙色区域
contours,h = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# 找到最小轮廓
a = []
for i in range(len(contours)):
    a.append(cv2.contourArea(contours[i]))
print(a)
min_index = np.argmin(a)
img_new= np.zeros(img.shape, np.uint8)
cv2.drawContours(img_new, contours, min_index, (255, 255, 255), -1)

# 通过掩码提取橙色区域
orange_region = cv2.bitwise_and(img, img_new)


# 显示结果
# cv2.imshow('Original Image', img)
cv2.imshow('Orange Region', orange_region)
cv2.waitKey()


cv2.imwrite("out8.jpg", orange_region)