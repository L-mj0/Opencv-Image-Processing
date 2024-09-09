import cv2
import numpy as np

# 读取图像
img = cv2.imread('9.jpg')

# 将图像从BGR颜色空间转换到HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义花瓣颜色的HSV范围
lower_pink = np.array([140, 0, 0])
upper_pink = np.array([170, 255, 255])

# 根据阈值构建掩码
mask = cv2.inRange(hsv, lower_pink, upper_pink)

# 使用形态学操作使颜色区域更加完整
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 通过掩码提取花瓣区域
flower_region = cv2.bitwise_and(img, img, mask=mask)


# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Mask', mask)
cv2.imshow('Flower Region', flower_region)
cv2.waitKey(0)

cv2.imwrite("out9.jpg", flower_region)