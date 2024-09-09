import cv2
import numpy as np


def gaussian_low_filter(shape, sigma):
    """ Gaussian Low Pass filter
	Args:
		shape: 滤波器形状
		sigma: sigma值（半径）
    H(u, v) = e^(-(u^2 + v^2) / (2 * sigma^2))
    """
    h, w = shape[: 2]
    cx, cy = int(w / 2), int(h / 2)
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    dis2 = u * u + v * v
    p = -dis2 / (2 * sigma**2)
    filt = np.exp(p)
    return filt


def answer1(img):
    img_caculator = img.copy()

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_caculator = cv2.filter2D(img_caculator, -1, kernel)

    # 提取边缘
    edges = cv2.Canny(img_caculator, 50, 150)

    # 寻找轮廓
    contours,h = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
            cv2.drawContours(img_caculator, contours, i, (0, 0, 255), 1)

    cv2.imshow('Market Image', img_caculator)
    cv2.imshow('Image', img)
    cv2.waitKey(0)

def answer2(img):
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # cv2.imshow('blurred', blurred)
    t, binary = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('binary', binary)
    # 形态学操作（开运算去除噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=5)

    # cv2.imshow('opening', opening)
    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # cv2.imshow('sure_bg', sure_bg)

    edges = cv2.Canny(opening, 50, 150)
    # enhanced_edges = cv2.convertScaleAbs(edges, alpha=500.0, beta=0)

    # cv2.imshow('enhanced_edges', enhanced_edges)
    # 查找轮廓
    contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    market = img2.copy()

    for i in range(len(contours)):
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)

        # 计算轮廓面积 (638, 188) (0, 437) (22, 457)
        area = cv2.contourArea(contours[i])
        if area > 4000:
            print(area)
            cv2.drawContours(market, contours, i, (0, 255, 0), 2)
        elif radius > 5:
            if center == (0, 437) or center == (22, 457):
                cv2.drawContours(market, contours, i, (0, 255, 0), 2)
            elif center != (638, 188):
                print(center)
                cv2.circle(market, center, radius, (0, 255, 0), 2)
                # 画圆心
                cv2.circle(market, center, 2, (0, 0, 255), 3)
    cv2.imshow('Market Image', market)
    
    cv2.imshow('Image', img2)
    cv2.waitKey(0)

        
     


# 读取图像
img1 = cv2.imread('6_1.tif',1)
img2 = cv2.imread('6_2.tif',1)

answer1(img1)
answer2(img2)

