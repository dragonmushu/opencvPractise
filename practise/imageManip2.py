#trying to retrieve profile of the receipt

import cv2
import numpy as np



img = cv2.imread('receipt1.jpg')
shape = img.shape
img = cv2.resize(img, (int(shape[0]/8), int(shape[1]/4)))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshImage = cv2.threshold(gray, 150, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
kernel = np.ones((25, 25))
dilate = cv2.dilate(threshImage, kernel)
dilate = cv2.erode(dilate, kernel)
threshImage2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0.2)

orig, contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
canny = cv2.Canny(threshImage, 105, 155)
cv2.imshow('thresh', gray)
cv2.imshow('dilate', dilate)
cv2.imshow('original', img)

