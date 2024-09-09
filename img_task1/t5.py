import cv2
import numpy as np

def getanswer2(img2):
    hsv_img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 50])
    upper_blue = np.array([180, 50, 200])

    lower_green = np.array([0, 0, 32])
    upper_green = np.array([180, 50, 33])

    mask2 = cv2.inRange(hsv_img, lower_green, upper_green)
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    hsv_img[mask>0,0] = 120
    hsv_img[mask>0,1] = 150
    hsv_img[mask>0,2] = hsv_img[mask>0,2] +50
    hsv_img[hsv_img>255] = 255

    hsv_img[mask2>0,0] = 60
    hsv_img[mask2>0,1] = 150
    hsv_img[mask2>0,2] = hsv_img[mask2>0,2] +50
    hsv_img[hsv_img>255] = 255

    blue_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    cv2.imwrite('1.jpg',blue_img)
    cv2.imshow('blue_img',blue_img)
    cv2.waitKey(0)


def getanswer1(img1):
    hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 50])
    upper_blue = np.array([180, 50, 200])

    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    hsv_img[mask>0,0] = 120
    hsv_img[mask>0,1] = 150
    hsv_img[mask>0,2] = hsv_img[mask>0,2] +50
    hsv_img[hsv_img>255] = 255

    blue_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('blue_img',blue_img)
    cv2.imwrite('2.jpg',blue_img)
    cv2.waitKey(0)



img1 = cv2.imread('5_1.tif',1)
img2 = cv2.imread('5_2.tif',1)

getanswer1(img1)
getanswer2(img2)


cv2.imshow('img1',img1)
cv2.imshow('img2',img2)

cv2.waitKey(0)