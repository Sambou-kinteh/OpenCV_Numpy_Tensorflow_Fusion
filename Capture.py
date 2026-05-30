import cv2 as cv
import numpy as np

from Frame import Frame
from Video import Video

class Capture:

    def __init__(self, source : int|str, window_name : str, video: Video, scale : float = 1):
        self.source = source
        self.windowName : str = window_name
        self.scale : float = scale
        self.captureInstance = cv.VideoCapture(source)
        self.toFrame = lambda src, window_name: Frame(src, window_name)
        self.KEYS = []

    @staticmethod
    def captureFrame(capturedFrame : Frame):
        capturedFrame.show()

    @staticmethod
    def getThreshold():
        return cv.getTrackbarPos("Threshold", "TrackBar")

    @staticmethod
    def getMethod():
        return cv.getTrackbarPos("Method", "TrackBar")

    @staticmethod
    def getWaitKeyBetweenFrames():
        return cv.getTrackbarPos("WaitKey", "TrackBar") + 1     # avoid indefinite waiting with 0

    @property
    def getNewCapture(self):
        return self.toFrame(self.captureInstance.read(), self.windowName)

    def close(self):
        pass

    def waitKey(self, frame : Frame):

        if not len(self.KEYS) == 0:

            if self.KEYS[0] == ord('h'):
                self.scale = self.scale + .10   # todo: frame.rescale(.10, True)

            elif self.KEYS[0] == ord('l'):
                if not self.scale < .2: self.scale = self.scale - .10

            elif self.KEYS[0] == ord('c'):
                self.captureFrame(frame)

            elif self.KEYS[0] == ord('q'):
                raise StopIteration

            self.KEYS.pop(0)

    def __next__(self):

        # todo self.refresh_frame = self.captureInstance.read()
        # override this function
        # isSuccessPrev, prev_frame = self.getNewCapture

        key = Frame.waitUserKey(5)
        if not key == 255: self.KEYS.append(key)
        self.waitKey()

    def __iter__(self):
        return self


if __name__ == "__main__":

    captureInstance = Capture(srcQuad, "CaptureModule",1.3)

    try:
        while True:
            captureInstance.__next__()

    finally:
        cv.destroyAllWindows()
        if hasattr(captureInstance, "captureInstance") and captureInstance.captureInstance.isOpened():
            print("Video stream closed")
            captureInstance.captureInstance.release()
