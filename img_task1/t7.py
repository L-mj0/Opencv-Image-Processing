import cv2

img = cv2.imread('7-1.jpg',0)
temp = cv2.imread('7-2.jpg',0)
th,tw = temp.shape
print(th,tw)

_, mask = cv2.threshold(temp, 50, 255, cv2.THRESH_BINARY_INV)

result = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF_NORMED,mask=mask)
minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(result)
print(minVal,maxVal,minLoc,maxLoc)
bottomright = (maxLoc[0]+tw,maxLoc[1]+th)

cv2.rectangle(img,maxLoc,bottomright,255,3)
# cv2.circle(img, (int(minLoc[0]+th/2),int(minLoc[1]+tw/2)-5),65 , 255, 2)

# cv2.imshow('img',img)
# cv2.imshow('temp', temp)
# cv2.waitKey()

cv2.imwrite("out7.jpg", img)