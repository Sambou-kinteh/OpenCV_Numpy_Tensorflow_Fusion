
import cv2 as cv
import numpy as np
import threading
import ctypes

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
    It is an encapsulated ndarray class that combines numpy and opencv functions (todo pytorch)
    for faster implementations of my exercises and for future projects


    """

    class __PyFrameArrayObject(ctypes.Structure):

        _fields_ = [                            # HEADER
            ("ob_refcnt",   ctypes.c_ssize_t),  # reference count
            ("ob_type",     ctypes.c_void_p),   # object type Frame

            ("data",        ctypes.c_void_p),   # raw pixel buffer
            ("nd",          ctypes.c_int),      # num dimensions
            ("dimensions",  ctypes.c_void_p),   # shape
            ("strides",     ctypes.c_void_p),   # strides
            ("base",        ctypes.c_void_p),   # base object for views
            ("descr",        ctypes.c_void_p),  # dtype descriptor
            ("flags",       ctypes.c_int),      # writable, own data ...
            # TODO LATER __dict__ type <c_void_p> to __slots__ for frame subclass
        ]

    CALLBACK_POINTS : Final = "points"
    CALLBACK_AMOUNT : Final = "amount"
    CALLBACK_CALLABLE : Final = "callable"
    CALLBACK_CALL : Final = "call"
    CALLBACK_CALL_PARAMS : Final = "call_params"
    CALLBACK_CALL_RETURN : Final = "call_return"

    __DOMAIN_TYPE_SPATIAL : Final = 0
    __DOMAIN_TYPE_FREQUENCY : Final = 1
    __DOMAIN_TYPE_TENSOR : Final = 2

    __CALL_BOUNDED = "bounded"

    __GETATTR_EXCEPTIONS_NAMES = "names"
    __GETATTR_EXCEPTIONS_TYPES = "types"
    __GETATTR_EXCEPTIONS : dict[str, list] = {
        __GETATTR_EXCEPTIONS_NAMES : ["copy", ],
        __GETATTR_EXCEPTIONS_TYPES : [Slider, ]
    }


    # TODO: mehrere Views and can immer neue Views erstellen mit der möglichkeit dies in sich zu speichern
    # TODO: each view should be refrenced with an index starting 0 - Main View
    # TODO: serializable functionality
    # TODO: dragable point that determine px distance to other defined points irt and a variant where the distances are shown on the connecting lines using self.nearest
    # TODO: toFloat, toUint8, toTensor, toFrequencyDomain, toSpatialDomain (Frame should have class property (without setter) DOMAIN_TYPE)
    # TODO: different functionalities based on domain type, can have multiple domain types but with constaints
    # TODO: normalize, zero center and whitening maybe
    # TODO: prepare non dst instances that are through away for garbage collection (nr. ref = 0)
    # TODO: __scale to a class property (think about it)
    # TODO: select points with angle (as in a direction or not)
    # TODO: solving the dtype buffer problem: set though field to float32 on creation and quantize when showing if domain_type not in quantized form
    def __new__(cls, src : str|ndarray, window_name : str, slider : Slider = None, updatable : bool = True) -> "Frame|ndarray":

        # read in data and cast to subclass
        __frame_obj = np.asarray(cv.imread(src) if isinstance(src, str) else src).view(cls)

        # custom attributes to new obj
        __frame_obj.updatable = updatable
        __frame_obj.window_name = window_name
        __frame_obj.slider = slider
        __frame_obj.__scale = 1.0
        __frame_obj.__domain_type = cls.__DOMAIN_TYPE_SPATIAL

        return __frame_obj

    def __init__(self, src : str|ndarray, window_name : str,  slider : Slider = None, updatable : bool = True):

        self.window_name = window_name
        self.updatable = updatable
        self.slider = slider
        self.__scale = 1.0
        self.__domain_type = self.__DOMAIN_TYPE_SPATIAL
        # cv.namedWindow(self.window_name)

    def __array_finalize__(self, obj):

        # called when new view of array is made
        # must handle none objs
        if obj is None: return          # direct construction
        self.updatable = getattr(obj, "updatable", True)
        self.window_name = getattr(obj, "window_name", " ")
        self.slider = getattr(obj, "slider", None)
        self.__scale = getattr(obj, "__scale", 1.0)
        self.__domain_type = getattr(obj, "__domain_type", self.__DOMAIN_TYPE_SPATIAL)

    # override next to implement a generator Frame
    def __next__(self, **kwargs): ...

    def __iter__(self) -> "Frame":
        return self

    def __call__(self, func : Callable[["Frame|ndarray", ...], "Frame|ndarray"], **kwargs) -> "Frame":
        """
        call frame to run a function (transformation) on the frame
        :param func: a python function that gives back a frame
        :param kwargs: the keyword arguments of the function
        :return: same instance of the object
        """
        assert callable(func), "Passed argument to call is not a function"
        if self.updatable:
            if not kwargs.get(self.__CALL_BOUNDED): call = func(self, **kwargs)
            else:
                kwargs.pop(self.__CALL_BOUNDED)
                call = func(**kwargs)

            assert isinstance(call, ndarray), f"Function {func.__name__} didn't return the expected type"

        # else: return func(self, **kwargs)       # todo polish it

        # MILESTONE (ONE FRAME, DO ALL): __refresh_frame successfully bypassed (DST solved)

        obj_struct = ctypes.cast(id(self), ctypes.POINTER(self.__PyFrameArrayObject)).contents
        # call_struct = ctypes.cast(id(call), ctypes.POINTER(self.__PyFrameArrayObject)).contents
        # isBufferHigher : bool = self.nbytes < call.nbytes

        # self_buffer = obj_struct.data   # freed during del to prevent leakage
        # call_buffer = call_struct.data

        # obj_struct.data, call_struct.data = call_buffer, self_buffer

        # if call.dtype != self.dtype:    # allign dtypes     TODO remodel dtype allignment

        # ctypes.pythonapi.Py_DecRef(ctypes.cast(obj_struct.descr, ctypes.py_object))
        # ctypes.pythonapi.Py_IncRef(ctypes.py_object(call.dtype))
        # obj_struct.descr = id(call.dtype)

        if call.shape != self.shape:    # allign the shapes of object in memory and new data into memery at same address

            dim_ssize_t_pointer = ctypes.cast(obj_struct.dimensions, ctypes.POINTER(ctypes.c_ssize_t))
            stride_ssize_t_pointer = ctypes.cast(obj_struct.strides, ctypes.POINTER(ctypes.c_ssize_t))

            new_shape = call.shape
            new_stride = call.strides

            obj_struct.nd = len(new_shape)
            for i in range(obj_struct.nd):
                dim_ssize_t_pointer[i] = new_shape[i]
                stride_ssize_t_pointer[i] = new_stride[i]

        self[:] = call
        del call

        return self

    def __getattribute__(self, item: str):

        attr = super().__getattribute__(item)       # get bounded method from ndarray
        if not item.endswith("__") and callable(attr) and item not in type(self).__dict__:    # intercept only when non dunder, callable and not in self

            isIntercept : bool = True
            if hasattr(attr, "__name__"): # method
                if attr.__name__ in self.__GETATTR_EXCEPTIONS[self.__GETATTR_EXCEPTIONS_NAMES]: isIntercept = False
            elif hasattr(attr, "__class__"): # class
                if type(attr) in self.__GETATTR_EXCEPTIONS[self.__GETATTR_EXCEPTIONS_TYPES]: isIntercept = False

            if isIntercept:

                def intercept(**kwargs):
                    kwargs.update([(self.__CALL_BOUNDED, True)])
                    return self(attr, **kwargs)
                return intercept

        return attr


    # todo METHOD DEPRECIATED
    def __refresh_frame(self, src : ndarray) -> "Frame":

        # if hasDst: return None
        obj = self.__new__(
            self.__class__,
            src,
            self.window_name,
            self.slider,
            self.updatable
        )
        # __new__ calls __...finalize__ directly
        return obj

    @property
    def domain_type(self):
        return self.__domain_type

    @property
    def normalized(self) -> "Frame":

        norm = lambda frame: (frame - frame.mean()) / frame.std()
        return self(norm)


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

    def toFloat(self) -> "Frame":
        # todo update dtype in memory
        to_float = lambda frame: (frame - frame.min()) / (frame.max() - frame.min())
        return self(to_float)

    def toUint8(self) -> "Frame":
        # todo update dtype in memory
        to_int = lambda frame: np.clip(frame * 255, 0, 255).astype(np.uint8)
        return self(to_int)

    def toFreuqencyDomain(self) -> "Frame": ...

    def toSpatialDomain(self) -> "Frame": ...


    def rescale(self, scale : float = 1, isIncremental : bool = False) -> "Frame":

        """
        This function scales a frame to a desired resolution with a scaler where 1 means no scaling and .5 means half the original size

        :param scale: scaler to scale with
        :param isIncremental: sets if the current scale should be replaced or just decremented (negative scale to increment)
        :return: returns a scaled frame
        """

        if self.updatable and (not scale == 1 and not isIncremental):
            self.__scale = scale if not isIncremental else self.__scale - scale
            dsize = (int(self.shape[1] * self.__scale), int(self.shape[0] * self.__scale))
            interpolation = cv.INTER_CUBIC if scale > 1 else cv.INTER_AREA
            if scale > 1: return self.__refresh_frame(cv.resize(self, dsize=dsize, interpolation=interpolation))
            return self(
                cv.resize,
                dsize=dsize,
                interpolation=interpolation
            )
        return self

    def draw_point(self, point : list, radius : int = 5, color: tuple[int, int, int] = (0, 0, 255), thickness: int = 3):

        """
        This function draws a circle around point(s) on frame

        :param point: sets of (x, y) coordinates that should be drawn on frame
        :param radius: radius of circle
        :param color: color of the circle(s)
        :param thickness: thickness of the circumference
        :return: returns back frame with the points
        """

        if not point: return self
        def __draw(_point): self(cv.circle, center=_point, radius=radius, color=color, thickness=thickness)

        np.vectorize(__draw, signature="(2) -> ()")(point)      # todo draw_point() = __vectorized_draw_point() in getattr
        return self

    def draw_line(self, line : list[int], color: tuple[int, int, int] = (0, 0, 255), thickness: int = 1):

        """
        This function draws a line through pt1 and pt2 on frame

        :param line: should have the form [pt1_x pt1_y, pt2_x pt2_y]
        :param color: color of the line(s)
        :param thickness: thickness of the line(s)
        :return: returns back frame with the line(s)
        """

        if not line: return self
        def __draw(_line):
            assert len(_line) == 4, "Invalid construction of line coordinates"
            self(cv.line, pt1=_line[:2], pt2=_line[2:], color=color, thickness=thickness, lineType=cv.LINE_AA)

        np.vectorize(__draw, signature="(4) -> ()")(line)      # todo draw_line() = __vectorized_draw_line() in getattr
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
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.__select_points, param=kwargs)

    def __select_points(self, event : int, x : int, y : int, flag : int, params : dict):

        points : list = params[self.CALLBACK_POINTS]
        if event == cv.EVENT_LBUTTONDBLCLK and flag == 1:             # only double left -> add

            amount = params[self.CALLBACK_AMOUNT][0]
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

        _callable : Callable|None = params.get(self.CALLBACK_CALLABLE)
        call : bool|None = params.get(self.CALLBACK_CALL)
        call_params : dict|None = params.get(self.CALLBACK_CALL_PARAMS)

        if _callable is not None and call is not None:
            assert isinstance(call, bool), "Callback flag call is not bool"
            if call: params.update({self.CALLBACK_CALL_RETURN: self(_callable, **call_params)})

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
        return cv.waitKey(ms) & 0xFF

    @staticmethod
    def waitUserKeyAsync() -> int:
        # return cv.waitKey(1) & 0xFF
        ...

    def preprocessing(self, **kwargs) -> None:
        # TODO stack of functions and parameters and new ones can be added at any time
        try:
            while True:
                self.__next__()

        except StopIteration:
            pass

        finally:
            cv.destroyAllWindows()

    def save(self):
        # TODO: should call serialize with the flag of only saving on disk
        ...

    def serialize(self):

        data = {
            "window_name": self.window_name,
            "scale": self.__scale,
            "updatable": self.updatable,
        }
        ...

    def deserialize(self):
        ...



if __name__ == "__main__":

    frameObj : Frame = Frame("../assignment_01/img.png", "Bild")
    # print(frameObj[:, :, :].setWindowName("Sambou Kinteh")(cv.cvtColor, code=cv.COLOR_BGR2RGB).toGray()[:, :].rescale(2).show(10000))
    #
    # frameObj.toGray()
    # # frameObj.waitUserKey(0)
    #
    # data = {
    #     Frame.CALLBACK_AMOUNT: [3],
    #     Frame.CALLBACK_POINTS: []
    # }
    # frameObj.set_mouse_callback(**data)
    # # frameObj.preprocessing = lambda x : print(x)
    # frameObj.__next__ = ...

    frameObj.toGray().show(1)
    # frameObj.astype(dtype=np.float32)
    # print("new dtype: ", frameObj.dtype)
    # print(frameObj.waitUserKey(0))

    # while True:
    #
    #     frameObj.show(1)
    #     if len(data[Frame.CALLBACK_POINTS]) == 3:
    #         print(data[Frame.CALLBACK_POINTS])
    #         frameObj.destroy()
    #         break


__all__ = [Frame]