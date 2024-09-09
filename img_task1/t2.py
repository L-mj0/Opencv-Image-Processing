import cv2
import numpy as np

def get_roi(image,x,y,w,h):
    return image[y:y+h,x:x+w]

def get_warp(img,p1,p2):
    h,w = img.shape[:2]
    # print(h,w)
    m_p = cv2.getPerspectiveTransform(p1,p2)
    out_p = cv2.warpPerspective(img,m_p,(w,h))
    return out_p

def get_answer1(img):
    # 提取感兴趣区域 
    x, y, w, h = 55, 80, 840, 250  # 你可以根据需要调整这些值
    roi = get_roi(img, x, y, w, h)
    h,w = roi.shape[:2]
    # 4 138
    p1 = np.float32([[0,0],[w-1,0],[w-1,h-1],[4,138]])
    p2 = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])

    out_p = get_warp(roi,p1,p2)
    # 图像增强
    out_p = cv2.filter2D(out_p, -1, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    out_p = cv2.GaussianBlur(out_p, (5, 5), 0)
    return out_p,roi

def get_answer2(img):
    # 90 40 603 434
    x, y, w, h = 90, 40, 516, 394  # 你可以根据需要调整这些值
    roi = get_roi(img, x, y, w, h)
    h,w = roi.shape[:2]
    # 4 138
    p1 = np.float32([[0,89],[w-1,0],[w-1,h-1],[0,346]])
    p2 = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])

    out_p = get_warp(roi,p1,p2)
    # 图像增强
    out_p = cv2.filter2D(out_p, -1, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    out_p = cv2.GaussianBlur(out_p, (5, 5), 0)
    return out_p,roi

# (50 80) (890 330)
img1 = cv2.imread('2_1.jpg')
img2 = cv2.imread('2_2.jpg')

out1,roi1 = get_answer1(img1)
out2,roi2 = get_answer2(img2)



# cv2.imwrite('roi.jpg',roi)
cv2.imshow('out1',out1)
cv2.imshow('out2',out2)
cv2.imshow('roi1', roi1)
cv2.imshow('roi2', roi2)
# cv2.imshow('image1',img1)
# cv2.imshow('image2',img2)
cv2.waitKey(0)