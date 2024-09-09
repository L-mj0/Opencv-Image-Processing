import cv2
import numpy as np

def getbottles():
    img_caculator = img1.copy()
    # 图像二值化处理 将图片除黑色以外的点全部变为白色

    for i in range(img_caculator.shape[0]):
        for j in range(img_caculator.shape[1]):
            if img_caculator[i][j] == 0:
                img_caculator[i][j] = 0
            else:
                img_caculator[i][j] = 255

    # 图像锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_caculator = cv2.filter2D(img_caculator, -1, kernel)

    # 提取边缘
    edges = cv2.Canny(img_caculator, 50, 150)

    # 寻找轮廓
    contours,h = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    print("检测出：{}个瓶子".format(len(contours)))
    
    cv2.imshow('edges', edges)

    cv2.waitKey(0)
    return len(contours)


def findminbottle():
    print(img1.shape)
    gray_image = img1.copy()
    
    # 将灰色的点变成白色
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i][j] == 0:
                gray_image[i][j] = 0
            elif gray_image[i][j] > 225:
                gray_image[i][j] = 0
            else:
                gray_image[i][j] = 255
    
    kernel = np.ones((10, 10), np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

    # 对 gray_image 进行形态学闭运算（平滑）
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, h = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在img中框出最小矩形高度最小的瓶子
    min_h = 100000
    min_contour = None

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        print(x, y, w, h)
        if h < min_h:
            min_h = h
            min_contour = contour
    print(min_h)
    
    # 绘制
    result_image = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    if min_contour is not None:
        x, y, w, h = cv2.boundingRect(min_contour)
        for i in range(y, y + h):
            for j in range(x, x + w):
                if gray_image[i, j] > 0:  # 只处理非黑色部分
                    result_image[i, j] = (0, img1[i, j], 0)  # 设置不同的绿色
        
        # cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('gray_image', gray_image)
    
    
    cv2.imshow('result_image', result_image)
    cv2.waitKey(0)




img1 = cv2.imread('1.tif',-1)

getbottles()
findminbottle()

cv2.imshow('Original image', img1)
cv2.waitKey(0)

