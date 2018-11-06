#attempt to section receipt using histograms

import cv2
import numpy as np
import matplotlib.pyplot as plt

MAX_WIDTH = 500
HORIZONTAL_MORPH_KERNEL = (1, 25)
MAX_NUMBER_SIZE = 70
MIN_NUMBER_SIZE = 10

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
    print(xMin)
    reducedBoxes = []
    for box in boxes:
        if len(np.where(box[:, 0] > xMin)[0]) == 4:
            reducedBoxes.append(box)

    #obtain heights of all boxes
    finalBoxes = []
    for box in reducedBoxes:
        w = np.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
        h = np.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
        print(w, h)
        if (MIN_NUMBER_SIZE < w < MAX_NUMBER_SIZE and MIN_NUMBER_SIZE < h < MAX_NUMBER_SIZE):
            finalBoxes.append(box)
            cv2.drawContours(img, [np.int0(box)], 0, (0, 255, 0), 1)
    
    
    cv2.imshow('trial', img)
    


extractNumbersColumn('receipt2.jpg')
