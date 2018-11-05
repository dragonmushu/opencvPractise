import cv2
import numpy as np


vid = cv2.VideoCapture(0)
edges = vid.read()[1]

while(True):
    _, frame = vid.read()
    blurFrame = cv2.medianBlur(frame, 19)
    grayScale = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(grayScale, cv2.CV_8U, edges, ksize=5)
    reverseEdges = cv2.bitwise_not(edges)
    cimage, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cv2.imshow('original', frame)
    cv2.imshow('edges', edges)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow('cimage', frame)
    
    #cv2.imshow('edges reverse', reverseEdges)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
