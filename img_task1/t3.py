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



def getans1(img1):
    img_caculator = img1.copy()
    # 图像二值化处理 只保留火柴根
    for i in range(img_caculator.shape[0]):
        for j in range(img_caculator.shape[1]):
            if img_caculator[i][j] < 160:
                img_caculator[i][j] = 0
            else:
                img_caculator[i][j] = 255

    edges = cv2.Canny(img_caculator, 50, 150)
    # 寻找轮廓
    contours,h = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print("检测出：{}个火柴根".format(len(contours)))
    return edges, img_caculator

def getanswer2(img2):
    w,h = img2.shape[:2]
    mask = gaussian_low_filter(img2.shape,13)

    # 空域转频域，并中心化
    f = np.fft.fft2(img2)
    fc = np.fft.fftshift(f)

    # 平滑操作
    fc = mask * fc

    # 频域到空域
    fe = np.fft.fftshift(fc)
    fout = np.fft.ifft2(fe)
    imgout = np.abs(fout).astype("uint8")
    cv2.imshow('imgout',imgout)

    img_caculator2 = imgout.copy()
    for i in range(img_caculator2.shape[0]):
        for j in range(img_caculator2.shape[1]):
            if img_caculator2[i][j] > 160:
                img_caculator2[i][j] = 255
            elif img_caculator2[i][j] < 50:
                img_caculator2[i][j] = img2[i][j]
            else:
                img_caculator2[i][j] = 255

    # 去噪
    img_caculator2 = cv2.medianBlur(img_caculator2, 5)

    edges = cv2.Canny(img_caculator2, 50, 150)
    # 寻找轮廓
    contours,h = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print("检测出：{}个火柴根".format(len(contours)+1))

    for i in range(len(contours)):
        # 计算轮廓面积
        area = cv2.contourArea(contours[i])
        if area < 30:
            continue
        cv2.drawContours(img2, contours, i, (255, 0, 0), 1)
        # print("火柴根{}面积为：{}".format(i, area))
    cv2.waitKey(0)
    return img2

def getanswer3(img3):
    w,h = img3.shape[:2]
    mask = gaussian_low_filter(img3.shape,13)

    # 空域转频域，并中心化
    f = np.fft.fft2(img3)
    fc = np.fft.fftshift(f)

    # 平滑操作
    fc = mask * fc

    # 频域到空域
    fe = np.fft.fftshift(fc)
    fout = np.fft.ifft2(fe)
    imgout = np.abs(fout).astype("uint8")
    # cv2.imshow('imgout',imgout)

    img_caculator2 = imgout.copy()
    for i in range(img_caculator2.shape[0]):
        for j in range(img_caculator2.shape[1]):
            if img_caculator2[i][j] > 160:
                img_caculator2[i][j] = 255
            elif img_caculator2[i][j] < 130:
                img_caculator2[i][j] = 0
            else:
                img_caculator2[i][j] = 255

    # 去噪
    img_caculator2 = cv2.medianBlur(img_caculator2, 5)

    edges = cv2.Canny(img_caculator2, 50, 150)
    # 寻找轮廓
    contours,h = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        # 计算轮廓面积
        area = cv2.contourArea(contours[i])
        cv2.drawContours(img3, contours, i, (255, 0, 0), 1)
    print('检测出：{}个药片'.format(len(contours)))
    
    return img3,img_caculator2

img1 = cv2.imread('3_1.tif',0)
img2 = cv2.imread('3_2.tif',-1)
img3 = cv2.imread('3_3.tif',-1)

edges,img_caculator = getans1(img1)
imgout2 = getanswer2(img2)
imgout3,caculator = getanswer3(img3)


cv2.imshow('edges', edges)
cv2.imshow('img_caculator',img_caculator)
cv2.imshow('caculator',caculator)
cv2.imshow('imgout2',imgout2)
cv2.imshow('imgout3',imgout3)
cv2.waitKey(0)