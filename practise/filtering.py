#module to test filtering using different color systems

import cv2
import numpy as np



#practise for only having blue color
def onlyBlue(frame):
    frame[:, :, 1] = 0
    frame[:, :, 2] = 0
    return frame

#extracts the color in bgr
def extractColor(frame, color):
    if (color < 0 and color > 2):
        raise ValueError('Invalid Color')
    for i in {0, 1, 2} - {color}:
        frame[:, :, i] = 0
    return frame


        

vid = cv2.VideoCapture(0)

while(True):
    _, frame = vid.read()
    extractColor(frame, 2)
    cv2.imshow('Filtering Colors', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break





vid.release()
cv2.destroyAllWindows()
