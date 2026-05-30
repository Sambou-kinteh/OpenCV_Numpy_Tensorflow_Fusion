
import cv2 as cv
from typing import Callable


class Slider:

    def __init__(self, window_name : str, params : dict[str, list]):

        self.window_name = window_name
        self.__original_params = params
        self.scale = []
        self.params : list[str] = []
        self.last_params : dict = {}

    def __call__(self):

        cv.namedWindow(self.window_name)
        for slider_key in self.__original_params:
            slider_range : list = self.__original_params[slider_key]
            self.add((slider_key, slider_range))

    def param(self, param) -> int:
        return cv.getTrackbarPos(param, self.window_name)

    def get_params(self) -> dict[str, int]:
        return dict([(param, self.param(param)) for param in self.params])

    # @params.setter
    # def params(self, value) -> None:
    #     assert isinstance(value, dict), "Datatype of Params not dict"
    #     self.params = value

    def add(self, param : tuple[str, list]) -> None:
        assert isinstance(param, tuple), "Invalid parameter data"
        self.params.append(param[0])
        cv.createTrackbar(param[0], self.window_name, param[1][0], param[1][1], lambda _ : _)

    def apply_param(self, param_meta : tuple[str, Callable]): # name, function
        pass

    def serialize(self):
        ...

    def deserialize(self):
        ...


__all__ = [Slider]
# ----------------Example param
# params_box_filter : dict = {
#     "box_ddepth" : [5, 10],
#     "box_ksize" : [3, 7],
#     "box_normalize" : [0, 1]
# }

