import cv2
import numpy as np


vid = cv2.VideoCapture(0)
edges = vid.read()[1]

while(True):
    _, frame = vid.read()
    blurFrame = cv2.medianBlur(frame, 13)
    grayScale = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(grayScale, cv2.CV_8U, edges, ksize=5)
    kernel = np.zeros((3, 3), np.uint8)
    kernel[:, 0] = 1
    edges = cv2.dilate(edges, kernel, iterations=100)
    #cv2.filter2D(edges, -1, kernel, edges)
    #sharpeningKernel = -1*np.ones((3, 3))
    #sharpeningKernel[1,1] = 9
    #cv2.filter2D(edges, -1, sharpeningKernel, edges)
    #edges = cv2.blur(edges, (9, 9))
    reverseEdges = cv2.bitwise_not(edges)
    cv2.imshow('original', frame)
    #cv2.imshow('grayscale', grayScale)
    cv2.imshow('edges', edges)
    cv2.imshow('edges reverse', reverseEdges)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
