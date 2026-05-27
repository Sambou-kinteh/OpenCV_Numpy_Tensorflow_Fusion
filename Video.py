
import cv2 as cv
import numpy as np

from MyHelpers.Frame import Frame
from MyHelpers.Capture import Capture
from MyHelpers.TrackBar import Slider


class Video:

    """
    Plan: todo Video will be like a record contraining all past frames and can be serialized
    """
    def __init__(self, window_name: str, frame: list[Frame], slider: Slider):
        ...

    def serialize(self):
        ...

    def deserialize(self):
        ...