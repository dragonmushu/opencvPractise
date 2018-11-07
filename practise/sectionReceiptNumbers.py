#attempt to section receipt using histograms

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

MAX_WIDTH = 500
HORIZONTAL_MORPH_KERNEL = (1, 25)
MAX_NUMBER_SIZE = 70
MIN_NUMBER_SIZE = 10
EXPANDED_WIDTH = 200
EROSION_KERNEL = (3, 3)

def extractNumbersColumn(receiptFile):
    #read image and resize
    img = cv2.imread(receiptFile)
    shape = img.shape
    scaleFactor = shape[1]/MAX_WIDTH
    img = cv2.resize(img, (int(shape[1]/scaleFactor), int(shape[0]/scaleFactor)))

    #grayscale and threshold
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshImage = cv2.threshold(grayScale, 100, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    #horizontal morph_open (dilation followed by erosion)
    kernel = np.ones(HORIZONTAL_MORPH_KERNEL)
    processedImage = cv2.morphologyEx(threshImage, cv2.MORPH_OPEN, kernel)
    #contour using rotated rectangles
    _, contours, _ = cv2.findContours(processedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    boxes = np.array([cv2.boxPoints(cv2.minAreaRect(c)) for c in contours])

    #find minXPositions of all rectangles
    minXPositions = [min(box[i][0] for i in range(0, 4)) for box in boxes]
    hist, bins = np.histogram(minXPositions, bins='auto')

    #find right most peak in histogram
    #TODO: Find better way to achieve peak of bounding box locations
    rightPeakIdx = len(hist) - 1
    for idx in range(len(hist) - 1, -1, -1):
        if ((idx + 1 >= len(hist) or hist[idx] > hist[idx + 1]) and (idx - 1 < 0 or hist[idx] > hist[idx - 1])):
            rightPeakIdx = idx
            break

    #extract x min and x max locations
    xMin = int(bins[rightPeakIdx])
    reducedBoxes = []

    #filter also looking at min and max dimensions
    for box in boxes:
        if len(np.where(box[:, 0] > xMin)[0]) == 4:
            w = np.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
            h = np.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
            if (MIN_NUMBER_SIZE < w < MAX_NUMBER_SIZE and MIN_NUMBER_SIZE < h < MAX_NUMBER_SIZE):
                reducedBoxes.append(box)
                cv2.drawContours(img, [np.int0(box)], 0, (0, 255, 0), 1)

    #return the image and the bounding boxes
    return (img, threshImage, reducedBoxes)
    


def extractFloatingPoint(img, boundingBoxes):

    #create mask (inverted) such that it includes text and white space around
    #borders on pytesseract can cause noise
    allFloatingPoints = []
    for box in boundingBoxes:
        mask = np.zeros((img.shape[0], img.shape[1]))
        cv2.fillConvexPoly(mask, np.array(box, 'int32'), 255)
        mask = np.array(mask, dtype='uint8')
        mask= cv2.bitwise_not(mask)
        isolatedText = cv2.bitwise_or(threshImage, mask)
        #section of image
        #find min y, max y, min x, max x
        xMax = int(max([box[i][0] for i in range(0, 4)]))
        yMax = int(max([box[i][1] for i in range(0, 4)]))
        xMin = int(min([box[i][0] for i in range(0, 4)]))
        yMin = int(min([box[i][1] for i in range(0, 4)]))
        isolatedText = isolatedText[yMin:yMax, xMin:xMax]
        #expand image
        shape = isolatedText.shape
        scaleFactor = shape[1]/EXPANDED_WIDTH
        resizedText = cv2.resize(isolatedText, (int(shape[1]/scaleFactor), int(shape[0]/scaleFactor)))
        #erode image to decrease holes
        kernel = np.ones(EROSION_KERNEL)
        textFilled = cv2.erode(resizedText, kernel)
        #TODO - affine the image so that the numbers are more straight
        
        #put image through tesseract and extract text
        cv2.imwrite('temp.jpg', textFilled)
        text = pytesseract.image_to_string(Image.open('temp.jpg'), config="--psm 13").strip()

        #process alpha numerical parts of text to conform to #.#
        #can be improved (need to analyze more receipts)
        value = ''
        for idx, c in enumerate(text):
            if (c.isdigit()):
                value += c
            elif (c == '-' and len(value) == 0):
                value += c
            elif (c == '.' and value.find(c)==-1):
                value += c
            elif ((c == '_' or c == '-' or c == ',') and value.find('.') == -1):
                value += '.'
            elif (len(value) - value.find('.') - 1 == 2):
                break
            elif (len(value) > 0):
                value = ''
                break
        if (len(value) != 0 and value.find('.')!=-1 and (len(value) - value.find('.') - 1 == 2)):
            try:
                allFloatingPoints.append(float(value.strip('""')))
            except ValueError:
                pass
                
    #cv2.imshow('mask', textFilled)
    return allFloatingPoints



img, threshImage, boxes = extractNumbersColumn('receipt7.png')
values = extractFloatingPoint(threshImage, boxes)
print(values)
cv2.imshow('image', threshImage)
print('Total: ', max(values))
