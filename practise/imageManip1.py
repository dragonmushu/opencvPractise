#
# attempt to retrieve total of receipt

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
img = cv2.imread('receipt1.jpg')
shape = img.shape
img = img[:, int(shape[1]/2):, :]
img = cv2.resize(img, (int(shape[1]/8), int(shape[0]/4)))


grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('grayScale', grayScale)

blur = cv2.medianBlur(grayScale, 5)
#cv2.imshow('median blur', blur)

threshImage = cv2.adaptiveThreshold(grayScale, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 1)
#cv2.imshow('threshold image', threshImage)


#blur = cv2.medianBlur(threshImage, 5)
#cv2.imshow('median blur', blur)

#gaussBlur = cv2.GaussianBlur(grayScale, (5, 5), 0)
#cv2.imshow('gauss blur', gaussBlur)

_, threshImage2 = cv2.threshold(grayScale, 100, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
#cv2.imshow('threshold Image Binary', threshImage2)


#dilation
kernel = np.ones((1, 25))
opening = cv2.morphologyEx(threshImage2, cv2.MORPH_OPEN, kernel)
cv2.imshow('dilation', opening)


#find contours
orig, contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

'''
rect = cv2.minAreaRect(contours[16])
box = cv2.boxPoints(rect)


w = np.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
h = np.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
d1 = np.array([0.2*(box[0][0] - box[1][0]), 0.2*(box[0][1] - box[1][1])])
d2 = np.array([0.2*(box[1][0] - box[2][0]), 0.2*(box[1][1] - box[2][1])])
box1 = box.copy()
box1[0] += (d1 + d2)
box1[1] += (-1*d1 + d2)
box1[2] += (-1*d1 + -1*d2)
box1[3] += (d1 + -1*d2)
xMax = int(max([box[i][0] for i in range(0, 4)]))+10
yMax = int(max([box[i][1] for i in range(0, 4)]))+10
xMin = int(min([box[i][0] for i in range(0, 4)]))-10
yMin = int(min([box[i][1] for i in range(0, 4)]))-10
print(xMax, yMax, yMin, xMin)
cv2.drawContours(img, [np.int0(box1)], 0, (0, 255, 0), 1)
mask = np.zeros((img.shape[0], img.shape[1]))
cv2.fillConvexPoly(mask, np.array(box1, 'int32'), 255)
#cv2.imshow('mask', mask)
mask =np.array(mask, dtype='uint8')
im1 = cv2.bitwise_and(threshImage2, mask)
mask = cv2.bitwise_not(mask)
im1 = cv2.bitwise_or(threshImage2, mask)
im2 = im1[yMin:yMax, xMin:xMax]
res = cv2.resize(im2, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
kernel = np.ones((3, 3))
res = cv2.erode(res, kernel)
cv2.imshow('temp', res)
cv2.imwrite('temp.jpg', res)
text = pytesseract.image_to_string(Image.open('temp.jpg'), config="--psm 13 outputbase digits")
os.remove('temp.jpg')
print('Text:', text)
'''

allValues = []
allText = []
for b in contours:
    #obtain rectangle and confining box
    rect = cv2.minAreaRect(b)
    box = cv2.boxPoints(rect)
    w = np.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
    h = np.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
    #exlude widths and heights less than 5 px
    if (w < 5 or h <5):
        continue
    
    # draw contours
    #cv2.drawContours(img, [np.int0(box)], 0, (0, 0, 255), 1)
    

    #expand box size to include edges
    d1 = np.array([0.1*(box[0][0] - box[1][0]), 0.1*(box[0][1] - box[1][1])])
    d2 = np.array([0.1*(box[1][0] - box[2][0]), 0.1*(box[1][1] - box[2][1])])
    box1 = box.copy()
    box1[0] += (d1 + d2)
    box1[1] += (-1*d1 + d2)
    box1[2] += (-1*d1 + -1*d2)
    box1[3] += (d1 + -1*d2)
    #find max x and y
    xMax = int(max([box[i][0] for i in range(0, 4)]))+10
    yMax = int(max([box[i][1] for i in range(0, 4)]))+10
    xMin = int(min([box[i][0] for i in range(0, 4)]))-10
    yMin = int(min([box[i][1] for i in range(0, 4)]))-10
    if (xMin < 0):
        xMin = 0
    if (yMin < 0):
        yMin = 0
    if (xMax > img.shape[1]):
        xMax = img.shape[1]
    if (yMax > img.shape[0]):
        yMax = img.shape[0]
    cv2.drawContours(img, [np.int0(box1)], 0, (0, 255, 0), 1)
    #mask all except current box
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(mask, np.array(box1, 'int32'), 255)
    mask =np.array(mask, dtype='uint8')
    im1 = cv2.bitwise_and(threshImage2, mask)
    mask = cv2.bitwise_not(mask)
    im1 = cv2.bitwise_or(threshImage2, mask)
    #resize image
    im2 = im1[yMin:yMax, xMin:xMax]
    res = cv2.resize(im2, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    #erode image to make more bold
    kernel = np.ones((3, 3))
    res = cv2.erode(res, kernel)
    #cv2.imshow('temp', res)
    #obtain text
    cv2.imwrite('temp.jpg', res)
    text = pytesseract.image_to_string(Image.open('temp.jpg'), config="--psm 13").strip()
    text = text.replace('-', '.')
    if len(text) > 0 and text[0] == '.':
        text = '-'+text[1:]
    allText.append(text)
    if (all([c.isdigit() or c=='.' or c == '-' for c in text])):
        try:
            allValues.append(float(text.strip('""')))
        except ValueError:
            pass     
    os.remove('temp.jpg')
print(allValues)
print(allText)

#cv2.drawContours(img, boxes, 0, (0, 0, 255), 3)

cv2.imshow('reciept', img)
