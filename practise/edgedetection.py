#edge detection module
import cv2
import numpy as np
import cvwindow

vid = cv2.VideoCapture(0)

while (True):
    _, frame = vid.read()
    #frame = cv2.medianBlur(frame, 9)
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(grayImage, cv2.CV_8U, grayImage, ksize=3)
    cv2.imshow('frame', frame)
    cv2.imshow('edges', grayImage)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
