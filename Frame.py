
import cv2 as cv
import numpy as np
import sys
import threading

from TrackBar import Slider        # relative import
from numpy import ndarray
from typing import Callable, Final


def run_async(runnable : Callable):

    # return_when_done = [1]
    def wmethod(*args, **kwargs):
        thread : threading.Thread = threading.Thread(target=runnable, args=[*args], kwargs={**kwargs})
        thread.name = runnable.__name__
        thread.start()
        # return return_when_done
    return wmethod


class Frame(ndarray):

    """
    Side Project:\n
    Frame inherits all native functions of ndarray.
    It is an encapsulated ndarray class that combines numpy and opencv functions
    for faster implementations of my exercises and for future projects

    DST Mode: Wenn eine Function kein DST hat, funktioniert self falls self.shape (was bei __new__ im objekt selbst festgelegt wird) nicht beeinflusst wird

    TODO:

    """

    CALLBACK_POINTS : Final = "points"
    CALLBACK_AMOUNT : Final = "amount"
    CALLBACK_CALLABLE : Final = "callable"
    CALLBACK_CALL : Final = "call"
    CALLBACK_CALL_PARAMS : Final = "call_params"
    CALLBACK_CALL_RETURN : Final = "call_return"


    # TODO: mehrere Views and can immer neue Views erstellen mit der möglichkeit dies in sich zu speichern
    # TODO: each view should be refrenced with an index starting 0 - Main View
    # TODO: serializable functionality
    # TODO: dragable point that determine px distance to other defined points irt and a variant where the distances are shown on the connecting lines using self.nearest
    # TODO: toFloat, toUint8, toFrequencyDomain, toSpatialDomain (Frame should have __<VARIABLE> with name DOMAIN_TYPE)
    def __new__(cls, src : str|ndarray, window_name : str, slider : Slider = None, updatable : bool = True) -> "Frame|ndarray":

        # read in data and cast to subclass
        _frame : ndarray = cv.imread(src) if isinstance(src, str) else src
        _frame_obj = np.asarray(_frame).view(cls)

        # add custom attributes to new obj
        _frame_obj._updatable = updatable
        _frame_obj._window_name = window_name
        _frame_obj._slider = slider
        _frame_obj._scale = 1.0

        return _frame_obj

    def __init__(self, src : str|ndarray, window_name : str,  slider : Slider = None, updatable : bool = True):

        self.window_name = window_name
        self.updatable = updatable
        self.slider = slider
        self.scale = 1.0
        cv.namedWindow(self.window_name)

    def __array_finalize__(self, obj):

        # called when new view of array is made
        # must handle none objs
        if obj is None: return          # direct construction
        self.updatable = getattr(obj, "_updatable", True)
        self.window_name = getattr(obj, "_window_name", " ")
        self.slider = getattr(obj, "_slider", None)
        self.scale = getattr(obj, "_scale", 1.0)

    def show(self, waitKey : None|int = None) -> None|int:
        cv.imshow(self.window_name, self)
        if waitKey is not None: return self.waitUserKey(waitKey)
        return None

    def destroy(self) -> None:
        """
        destroys all active windows
        :return: None
        """

        cv.destroyWindow(self.window_name)

    def setWindowName(self, window_name : str) -> "Frame":
        """
        Gives frame a new window name and calls cv.namedWindow on it
        :param window_name: new window name for frame
        :return: returns frame instance with new window name
        """

        cv.namedWindow(window_name)
        self.window_name = window_name
        return self

    def toGray(self) -> "Frame":
        """
        :return: gray scale version of frame
        """

        return self(cv.cvtColor, code=cv.COLOR_BGR2GRAY)

    def rescale(self, scale : float = 1, isIncremental : bool = False) -> "Frame|ndarray":

        """
        This function scales a frame to a desired resolution with a scaler where 1 means no scaling and .5 means half the original size

        :param scale: scaler to scale with
        :param isIncremental: sets if the current scale should be replaced or just decremented (negative scale to increment)
        :return: returns a scaled frame
        """

        if self.updatable and (not scale == 1 and not isIncremental):
            self.scale = scale if not isIncremental else self.scale - scale
            return self(
                cv.resize,
                dsize=(int(self.shape[1] * self.scale), int(self.shape[0] * self.scale)),
                interpolation=cv.INTER_CUBIC if scale > 1 else cv.INTER_AREA
            )
        return self

    def draw_point(self, point, radius : int = 5, color: tuple[int, int, int] = (0, 0, 255), thickness: int = 3):

        """
        This function draws a circle around point(s) on frame

        :param point: sets of (x, y) coordinates that should be drawn on frame
        :param radius: radius of circle
        :param color: color of the circle(s)
        :param thickness: thickness of the circumference
        :return: returns back frame with the points
        """

        if not point: return self
        def __draw(points): self(cv.circle, center=points, radius=radius, color=color, thickness=thickness)

        np.vectorize(__draw, signature="(2) -> ()")(point)      # todo draw_point() = __vectorized_draw_point()
        return self


    def set_mouse_callback(self, **kwargs) -> None:
        """
        Callback function for on-click events on Frame, based on OpenCV cv.setMouseCallback.
        \nFunctionalities:
        \n- double (left) click to add a point
        \n- double (left) click + shift to remove a point in the area

        :param kwargs: must have kwargs: amount : list, points : list, TODO type
        :return: None
        """

        # TODO implement type with CALLBACK_TYPE_<POINTS, AREA>
        # TODO implement min_distance allowed between points as optional parameter
        # TODO allow user to set custom callback function but through a standard pipeline
        assert kwargs.get(self.CALLBACK_AMOUNT) is not None and isinstance(kwargs.get(self.CALLBACK_AMOUNT), list), "Mouse callback must have kwarg: amount"
        assert kwargs.get(self.CALLBACK_POINTS) is not None and isinstance(kwargs.get(self.CALLBACK_POINTS), list), "Mouse callback must have kwarg: points"
        cv.setMouseCallback(self.window_name, self.__select_points, param=kwargs)

    def __select_points(self, event : int, x : int, y : int, flag : int, params : dict):

        points : list = params[Frame.CALLBACK_POINTS]
        if event == cv.EVENT_LBUTTONDBLCLK and flag == 1:             # only double left -> add

            amount = params[Frame.CALLBACK_AMOUNT][0]
            if len(points) < amount:
                if len(points) > 0:
                    _, distances = self.nearest_point((x, y), points)
                    # todo if point in array or in close prorimity dont add, for now 20px, later additional optional argument
                    if distances[0] < 20:
                        print("[INFO] point already selected")
                    else:
                        points.append((x,y))
                        print(f"[INFO] current points {len(points)}")
                else:
                    points.append((x,y))
                    print(f"[INFO] current points {len(points)}")

        elif event == cv.EVENT_LBUTTONDBLCLK and flag & cv.EVENT_FLAG_SHIFTKEY:     # double left with shift -> remove
            if len(points) > 0:
                i, distances = self.nearest_point((x, y), points)
                if distances[0] < 20:
                    points.pop(i)
                    print(f"[INFO] current points after removal: {len(points)}")
                else: print(f"[INFO] no points in range")

        _callable : Callable|None = params.get(Frame.CALLBACK_CALLABLE)
        call : bool|None = params.get(Frame.CALLBACK_CALL)
        call_params : dict|None = params.get(Frame.CALLBACK_CALL_PARAMS)

        if _callable is not None and call is not None:
            assert isinstance(call, bool), "Callback flag call is not bool"
            if call: params.update({Frame.CALLBACK_CALL_RETURN: self(_callable, **call_params)})

    # TODO: implement
    def __select_area(self, event : int, x : int, y : int, flag : int, params : dict): ...

    @staticmethod
    def nearest_point(point : tuple, points : list[tuple[int, int]], flag : int = 0) -> tuple|None:

        """
        Checks the distance from a point to all other points
        :param point: point to check distance of
        :param points: all the points to be tested against
        :param flag: 0 for value sorting and 1 for index sorting, else returns None
        :return: returns index of the point with the min distance, and a sorted list (ascending) of the distance values
        """
        #TODO vectorize func

        px_distances = np.linalg.norm(np.array(point) - np.array(points), axis=1)   # broadcasts point to N x 2 and substracts from N x 2
        if flag == 0: return np.argmin(px_distances), np.sort(px_distances)
        elif flag == 1: return np.argmin(px_distances), np.argsort(px_distances)
        return None

    @staticmethod
    def waitUserKey(ms : int) -> int:
        """
        a key on-click listener with timeout-timer in milliseconds
        :param ms: timer in ms
        :return: returns key pressed, if nothing was registered, returns 255
        """
        assert isinstance(ms, int), "WaitKey timeout must always be an integer in milliseconds"
        key = cv.waitKey(ms) & 0xFF
        return key

    @staticmethod
    def waitUserKeyAsync() -> int: return cv.waitKey(1) & 0xFF

    def preprocessing(self, **kwargs) -> None:
        # TODO stack of functions and parameters and new ones can be added at any time
        try:
            while True:
                self.__next__()

        except StopIteration:
            pass

        finally:
            cv.destroyAllWindows()

    # override next to implement a generator Frame
    def __next__(self, **kwargs): ...

    def __iter__(self) -> "Frame":
        return self

    def __call__(self, func : Callable[["Frame|ndarray", ...], "Frame|ndarray"], **kwargs) -> "Frame":
        """
        call frame to run a function (transformation) on the frame
        :param func: a python function that gives back a frame
        :param kwargs: the keyword arguments of the function
        :return:
        """

        assert isinstance(func, Callable), "Passed argument to call is not a function"
        if self.updatable:
            try:
                call = func(self, dst=self, **kwargs)
            except Exception:
                exceptions = ["line", "circle"]
                if not func.__name__ in exceptions:
                    print(f"[WARNING] DST mode unavailable, make sure to store the object for which the function <{func.__name__}> was used on", file=sys.stderr)
                call = func(self, **kwargs)

            assert isinstance(call, ndarray), f"Function {func.__name__} didn't return the expected type"
        return self.__refresh_frame(call)

    def __refresh_frame(self, src : str | ndarray) -> "Frame":

        # if hasDst: return None
        obj = self.__new__(
            self.__class__,
            src,
            self.window_name,
            self.updatable
        )
        obj.__array_finalize__(self)        # finalize so eine neue View auf Klasse nicht nötig
        return obj

    def save(self):
        # TODO: should call serialize with the flag of only saving on disk
        ...

    def serialize(self):

        data = {
            "window_name": self.window_name,
            "scale": self.scale,
            "updatable": self.updatable,
        }
        ...

    def deserialize(self):
        ...



if __name__ == "__main__":

    frameObj : Frame = Frame("../assignment_01/img.png", "Bild")
    # print(frameObj[:, :, :].setWindowName("Sambou Kinteh")(cv.cvtColor, code=cv.COLOR_BGR2RGB).toGray()[:, :].rescale(2).show(10000))

    data = {
        Frame.CALLBACK_AMOUNT: [3],
        Frame.CALLBACK_POINTS: []
    }
    frameObj.set_mouse_callback(**data)
    # frameObj.preprocessing = lambda x : print(x)
    frameObj.__next__ = ...

    while True:

        frameObj.show(1)
        if len(data[Frame.CALLBACK_POINTS]) == 3:
            print(data[Frame.CALLBACK_POINTS])
            frameObj.destroy()
            break


__all__ = [Frame]