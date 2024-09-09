import cv2
import numpy as np

def getanswer(img):
    # 直方图均衡化以增强对比度
    equalized = cv2.equalizeHist(img)

    # 高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # 自适应阈值二值化
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学操作以闭合断开的地方
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 提取边缘以进一步闭合断裂
    edges = cv2.Canny(closed_image, 50, 150)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 合并边缘和闭合的图像
    out = cv2.bitwise_or(closed_image, edges_dilated)

    # 显示结果

    # cv2.imshow('Equalized Image', equalized)
    # cv2.imshow('Binary Image', binary_image)
    # cv2.imshow('Closed Image', closed_image)
    # cv2.imshow('Final Enhanced Image', out)
    # cv2.waitKey(0)

    cv2.imwrite('equalized_image.jpg', equalized)
    cv2.imwrite('binary_image.jpg', binary_image)
    cv2.imwrite('closed_image.jpg', closed_image)
    return out


# 读取图像
img = cv2.imread('4.tif', 0)

out = getanswer(img)

# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
cv2.imwrite('out.jpg', out)
