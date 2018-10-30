import cv2
import numpy as np


class Window:
    def __init__ (self, name):
        self._name = name
        self._isWindowActive = false

    def createWindow(self):
        if self._isWindowActive:
            raise RuntimeError
        cv2.namedWindow(self._name)
        self._isWindowActive = true

    def show(self, frame):
        if not self._isWindowActive:
            raise RuntimeError
        cv2.imshow(self._name, frame)
        
    def close(self):
        if not self._isWindowActive:
            raise RuntimeError
        cv2.destroyWindow(self._name)
        self_isWindowActive = false

    
        
