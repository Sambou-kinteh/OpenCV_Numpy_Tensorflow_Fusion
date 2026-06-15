
import numpy as np
import cv2 as cv
import os

from numpy import ndarray
from Frame import Frame


"""
Functionalities:

- adding points by (left) double click
- removing points by (left) double click + shift 
- removing all points at once by pressing "d"
- closing all frames at once by pressing "q"
- see Frame documentation for more details
"""

def load_images(path : str = "squirrel_images_and_data", mats : list[int] = None) -> list:

    if not mats:        # if no numbers provided, take two unique random numbers and work with that
        rand1 = np.random.randint(0, 36)
        rand2 = np.random.randint(0, 36)
        while rand1 == rand2:
            rand2 = np.random.randint(0, 36)

    else:
        assert len(mats) == 2
        rand1, rand2 = mats

    matrix1_name = f"viff{0}{0 if rand1 < 10 else ""}{rand1}_matrix"
    matrix2_name = f"viff{0}{0 if rand2 < 10 else ""}{rand2}_matrix"
    K_name = "K_matrix"

    frame1_name = os.path.join(path, f"image_{rand1}.jpg")
    frame2_name = os.path.join(path, f"image_{rand2}.jpg")

    f_viff = cv.FileStorage(os.path.join(path, "viff.xml"), cv.FILE_STORAGE_READ)
    f_K = cv.FileStorage(os.path.join(path, "K.xml"), cv.FILE_STORAGE_READ)

    P1 = np.asarray(f_viff.getNode(matrix1_name).mat())
    P2 = np.asarray(f_viff.getNode(matrix2_name).mat())
    K = np.asarray(f_K.getNode(K_name).mat()).reshape((3, 3))

    f_viff.release()
    f_K.release()

    return [
        [Frame(frame1_name, "Frame 1"), Frame(frame2_name, "Frame 2")],
        [P1, P2],
        K
    ]

def determine_extrinsics(P : ndarray, _K : ndarray) -> tuple[ndarray, ndarray]:

    M = P[:, :-1]

    # P = [KR, -KRc] = [KR, Kt]
    R = np.linalg.inv(_K) @ M
    t = np.linalg.inv(_K) @ P[:, -1]

    return R, t

def determine_E_using_P(Ps : list[ndarray], K : ndarray) -> tuple[ndarray, ndarray]:

    # frame 2 is defined in reference to frame 1
    # since frame 1 is not defined as (0, 0, 0) and most likely has a K != R != I

    R1, t1 = determine_extrinsics(Ps[0], K)
    R2, t2 = determine_extrinsics(Ps[1], K)

    # maps from 1 to 2
    R_2_1 : ndarray = R2 @ R1.T           # correct the coordinate system P1 to I = R1 then rotate to R2    # R21 = R2 if R1=I
    t_2_1 : ndarray = t2 - (R_2_1 @ t1)   # rotate center of P1 relative to P2 then calculate the baseline  # t21 = t2 if t1=0

    T : ndarray = np.zeros((3, 3))
    T[0, 1] = -t_2_1[-1]
    T[0, 2] = t_2_1[-2]
    T[1, 2] = -t_2_1[-3]
    T = T - T.T

    return (T @ R_2_1).T, R_2_1


def determine_lines(_F: ndarray, _points: list[list], _lines : list[dict[tuple, list]]):
    # for every point in points, check if in lines, if not, determine it and add to lines, else go further
    # also check if point in lines in points, then delete line: meaning user deleted the point

    # forward is frame 2 to frame 1
    transform = lambda _F, _point, forward = True: (
            (_F if forward else _F.T) @ (list(_point) + [1])
    )

    for i in range(len(_points)):
        for j in range(len(_points[i])):

            point = _points[i][j]
            line = _lines[i].get(point)
            if not line: line = transform(_F, point, i == 1)
            else: continue

            _lines[i].update([(point, determine_line_coordinates(line))])

    for i in range(len(_lines)):
        for _key in list(_lines[i].keys()):
            if _key not in _points[i]: _lines[i].pop(_key)


def determine_line_coordinates(_line, img_width = 640, img_height = 480) -> list:


    lx, ly, lw = _line

    epsilon = 1e-6

    if abs(ly) > epsilon:
        # x=0 and x=img_width to find corresponding y values.
        x1 = 0
        y1 = int(round((-lw - lx * x1) / ly))

        x2 = img_width
        y2 = int(round((-lw - lx * x2) / ly))
    else:
        # y=0 and y=img_height to find corresponding x values.
        y1 = 0
        x1 = int(round((-lw - ly * y1) / lx))

        y2 = img_height
        x2 = int(round((-lw - ly * y2) / lx))

    return [x1, y1, x2, y2]

def determine_axis_angle(rot : ndarray) -> tuple[ndarray, float]:

    r_vec, _ = cv.Rodrigues(rot)
    _angle = np.linalg.norm(r_vec)

    if _angle == 0: return None, None
    return (r_vec / _angle).flatten(), np.degrees(_angle)

if __name__ == '__main__':

    scale = .5      # assumption: scaling image once and not a trackbar (not explizitly defined)
    frames, camera_matrices, K = load_images(mats=[0, 14]) # mats=[0, 14] to run example on excersice
    frames = [frame.rescale(scale) for frame in frames]     # rescale with a scaling factor

    #------------ variables to keep track to store the points and lines for each image ---------
    allowed_num_of_points = np.inf

    data_frame1 = {
        Frame.CALLBACK_POINTS: [],
        Frame.CALLBACK_AMOUNT: [allowed_num_of_points],
    }
    data_frame2 = {
        Frame.CALLBACK_POINTS: [],
        Frame.CALLBACK_AMOUNT: [allowed_num_of_points],
    }

    # (point in img_x, line in img_y)
    lines_frame1 : dict[tuple, list] = {}
    lines_frame2 : dict[tuple, list] = {}

    #------------ E ------------
    E, R = determine_E_using_P(camera_matrices, K)
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    axis, angle = determine_axis_angle(R)
    print("Axis: ", axis)
    print("Angle: ", angle)

    frames[0].set_mouse_callback(**data_frame1)
    frames[1].set_mouse_callback(**data_frame2)

    isStage : bool = False

    while True:

        #------------ preprocessing ----------------
        if not isStage:
            frames[0].show(1)
            frames[1].show(1)
        key = Frame.waitUserKey(3)

        determine_lines(
            F,
            [data_frame1[Frame.CALLBACK_POINTS], data_frame2[Frame.CALLBACK_POINTS]],
            [lines_frame2, lines_frame1]
        )   # only calculate new line if line of a point wasn't calculated, also when key in list delete key

        #------------ on-click control flow -------------------
        if key == ord('d'):

            data_frame1[Frame.CALLBACK_POINTS].clear()
            data_frame2[Frame.CALLBACK_POINTS].clear()

            lines_frame1.clear()
            lines_frame2.clear()

        elif key == ord('q'):
            cv.destroyAllWindows()
            break

        #------------ stage 0 (draw the points and lines) -------------------
        if data_frame1[Frame.CALLBACK_POINTS] or data_frame2[Frame.CALLBACK_POINTS] or isStage:

            #------------ draw points -------------
            kwargs = {
                "radius": 3,
                "color": (255, 0, 0),   # bgr
                "thickness": 2,
            }
            frame1_copy = frames[0].copy()
            frame2_copy = frames[1].copy()

            frame1_copy.draw_point(data_frame1[Frame.CALLBACK_POINTS], **kwargs).show()
            kwargs["color"] = (0, 255, 255)
            frame2_copy.draw_point(data_frame2[Frame.CALLBACK_POINTS], **kwargs).show()

            #------------ draw lines -----------
            kwargs.pop("radius")
            frame1_copy.draw_line(list(lines_frame1.values()), **kwargs).show()
            kwargs["color"] = (255, 0, 0)
            frame2_copy.draw_line(list(lines_frame2.values()), **kwargs).show()

            isStage = True


