
import cv2 as cv
from typing import Callable


class Slider:

    def __init__(self, window_name : str, params : dict[str, list]):

        """
        A trackbar class based on the opencv cv.createTrackbar function
        :param window_name: Name of the window for the Trackbar, can be same as frame or different
        :param params: Keyword dict of all parameters of the trackbar. Every keys should be same as the keyword arguments for thier corresponding functions. Each key takes a list as value that has two values, start value and number of possible values starting from 0
        """

        self.window_name = window_name
        self.__original_params = params
        self.scale = []
        self.params : list[str] = []
        self.last_params : dict = {}

    def __call__(self) -> None:

        """
        initiates (starts) the slider
        :return: None
        """

        cv.namedWindow(self.window_name)
        for slider_key in self.__original_params:
            slider_range : list = self.__original_params[slider_key]
            self.add((slider_key, slider_range))

    def param(self, param) -> int:

        """
        :param param: parameter name
        :return: current value of a single parameter
        """
        assert param in self.params, "Invalid parameter for slider"
        return cv.getTrackbarPos(param, self.window_name)

    def get_params(self) -> dict[str, int]:

        """
        :return: returns list of all current parameter value pairs
        """
        return dict([(param, self.param(param)) for param in self.params])

    # @params.setter
    # def params(self, value) -> None:
    #     assert isinstance(value, dict), "Datatype of Params not dict"
    #     self.params = value

    def add(self, param : tuple[str, list]) -> None:

        """
        adds a single parameter to the slider instance. new parameters can be added at any time
        :param param: parameter name and range tuple, see class definition for more explanation
        :return: None
        """
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

