#learning blurring approaches
import cv2
import numpy as np

def convolutionBlurring(frame):
    kernel = 1/25*np.ones((5, 5))
    kernel[2, 2] = 0
    return cv2.filter2D(frame, -1, kernel)


def regularBlur(frame):
    return cv2.blur(frame, (5, 5))

def gaussianBlur(frame):
    return cv2.GaussianBlur(frame, (5, 5), 5)

def medianBlur(frame):
    return cv2.medianBlur(frame, 9)


vid = cv2.VideoCapture(0)


while(True):
    _, frame = vid.read()
    frame1 = gaussianBlur(frame)
    frame2 = regularBlur(frame)
    frame3 = medianBlur(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)
    cv2.imshow('frame3', frame3)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
