
import cv2 as cv
import numpy as np
from Frame import Frame
from TrackBar import Slider
from numpy import ndarray


"""
Workflow:

points will be determined by goodFeaturesToTrack but can be adjusted as well
if satisfied press c to continue with stage 2

Stage 2:
in stage two you select 2 points (corners) on two on the squares. see example picture for best results (to get the sequence is correct)
after selecting one, frame pauses to show which corners are allocated to that square
select a second point and then press a key to show the next 4 points of that 
the last 4 points will be allocated directly to the last square and pauses for 2s and goes to stage 3

Stage 3: 
in stage 3 K will be printed out and angle from projection center between two corners of the squares
"""


def determine_good_points(_frame: Frame, _data: dict = None, flag: int = 0) -> ndarray | None:

    if flag == 1:

        params = _frame.slider.get_params()
        umrechnen(params)
        if _frame.slider.last_params == params: return None
        else:
            _good_points : ndarray = cv.goodFeaturesToTrack(_frame, maxCorners=12, **params)
            _frame.slider.last_params = params

    else : _good_points : ndarray = cv.goodFeaturesToTrack(_frame, maxCorners=12, **_frame.slider.last_params)

    if _good_points is None: return None
    _good_points = _good_points.reshape(-1, 2).astype(int)

    if _data is None: return _good_points
    else:
        # new sets of points without changing the object reference in memory, list "equal to" new_list
        _data[Frame.CALLBACK_POINTS][:] = _good_points.tolist()
        # _data[Frame.CALLBACK_POINTS].clear()
        # _data[Frame.CALLBACK_POINTS].extend(_good_points.tolist())
        return None


def umrechnen(_params : dict) -> None:

    quality = _params["qualityLevel"]
    _params["qualityLevel"] = quality / 100 if not quality == 0 else .01
    size = _params["blockSize"]
    if size == 0:
        _params["blockSize"] = 1
        return
    _params["blockSize"] = size - 1 if size % 2 == 0 else size


def determine_H(points):

    # DLT
    # H * (0, 0, 1) = corresponding image point
    # H einheits recteck auf ebene ab
    H : ndarray = np.zeros((3, 3))

    x = lambda i : points[i - 1][0]
    y = lambda i : points[i - 1][1]

    nenner = (x(2) - x(3)) * (y(4) - y(3)) - (x(4) - x(3)) * (y(2) - y(3))
    H31 = ((x(1) - x(2) + x(3) - x(4)) * (y(4) - y(3)) - (y(1) - y(2) + y(3) - y(4)) * (x(4) - x(3))) / nenner
    H32 = ((y(1) - y(2) + y(3) - y(4)) * (x(2) - x(3)) - (x(1) - x(2) + x(3) - x(4)) * (y(2) - y(3))) / nenner

    H[0, 0] = x(2) - x(1) + H31 * x(2)
    H[0, 1] = x(4) - x(1) + H32 * x(4)
    H[0, 2] = x(1)

    H[1, 0] = y(2) - y(1) + H31 * y(2)
    H[1, 1] = y(4) - y(1) + H32 * y(4)
    H[1, 2] = y(1)

    H[2, 0] = H31
    H[2, 1] = H32
    H[2, 2] = 1

    return H


def determine_IAC(Hs : list[ndarray]) -> tuple[ndarray, ndarray]:

    # es gilt hi_1.T * w * hi_2 = 0
    Hi_1 : ndarray = np.zeros((3, 3), dtype=np.float64)
    Hi_2 : ndarray = np.zeros((3, 3), dtype=np.float64)

    # fill by rows
    Hi_1[0, :] = Hs[0][:, 0]
    Hi_1[1, :] = Hs[1][:, 0]
    Hi_1[2, :] = Hs[2][:, 0]

    # fill by columbs
    Hi_2[:, 0] = Hs[0][:, 1]
    Hi_2[:, 1] = Hs[1][:, 1]
    Hi_2[:, 2] = Hs[2][:, 1]

    # es gilt Hi_1 * w * Hi_2 = 0
    # Hi_1 * w * Hi_1.T = Hi_2.T * w * Hi_2
    # Hi_1 * w = Hi_2.T * w * Hi_2 * Hi_1.-T
    # => Hi_2.T * w * Hi_2 * Hi_1.-T * Hi_2 = 0         diese Gleichung vereint die beiden constraints

    v = np.linalg.svd(Hi_2.T, full_matrices=True)[2][-1, :]
    V = np.zeros((3, 3))
    V[:, 0] = V[:, 1] = V[:, 2] = v

    # w * Hi_2 * Hi_1.-T * Hi_2 = v_boardcasted = V, Hi_2.T * Hi_1.-1 * Hi_2.T * w.T = V.T
    w, res, rk, s = np.linalg.lstsq(Hi_2.T @ np.linalg.inv(Hi_1) @ Hi_2.T, V.T, rcond=None)

    # w = (KK.T)_inv, w.T = K.-T * K.-1, w.-T = K.T * K
    KtK = np.linalg.inv(w.T)
    # print(np.linalg.eigvals(KtK))  # hat negativer EW und auch schlect konditioniert

    return np.linalg.cholesky(KtK).T, w.T


def determine_IAC_standard(Hs : list[ndarray]) -> tuple[ndarray, ndarray]:

    # Zhangs (closed-form) Methode
    # B = (K.T * K)^-1, B ist symmetrisch also B ist definiert durch eine Dreiecksmatrix, was als 6D Vektor b dargestellt werden kann
    # b = [B11, B12, B22, B13, B23, B33].T
    # es gilt dann hi.T * w * hj = vij.T * b
    # vij = [hi1 * hj1, hi1 * hj2 + hi2 * hj1, hi2 * hj2, hi1 * hj3 + hi3 * hj1, hi2 * hj3 + hi3 * hj2, hi3 * hj3], wobei hi, hj aus Spalten von H sind
    # => matrix mit den constraints wäre [v12.T, (v11 - v22).T] =: V aus R(2nx6) für n Homograpien, also gilt V * b = 0
    # in diesem fall V ist 6x6

    vij = lambda i, j, H : np.array([
        H[:, i][0] * H[:, j][0],
        H[:, i][0] * H[:, j][1] + H[:, i][1] * H[:, j][0],
        H[:, i][1] * H[:, j][1],
        H[:, i][0] * H[:, j][2] + H[:, i][2] * H[:, j][0],
        H[:, i][1] * H[:, j][2] + H[:, i][2] * H[:, j][1],
        H[:, i][2] * H[:, j][2]
    ])

    V : ndarray = np.zeros((6, 6), dtype=np.float64)
    for i in range(0, 6, 2):
        V[i, :] = vij(0, 1, Hs[i//2])
        V[i+1, :] = vij(0, 0, Hs[i//2]) - vij(1, 1, Hs[i//2])

    b = np.linalg.svd(V, full_matrices=True)[2][-1, :]
    B = np.array([                                          # B = (K.T * K)^-1 = K.-T * K.-1
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]],
    ])

    B = B if not B[0, 0] < 0 else -B

    # print(np.linalg.eigvals(B))     # ebensfalls negativer EW drin, meistens
    # L = np.linalg.cholesky(B)       # gibt L aus bei L * L.T, wobei L untere Dreiecksmatrix, B = K.-T * K.-1 also L = K.-T
    # return np.linalg.inv(L.T), B

    # analytische extration von zhang
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1]**2)
    lam = B[2, 2] - (B[0, 2]**2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]

    alpha = np.sqrt(lam / B[0, 0])
    beta = np.sqrt(lam * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1]**2))
    gamma = -B[0, 1] * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B[0, 2] * alpha**2 / lam

    # Intrinsische Kameramatrix konstruieren
    _K = np.array([
        [alpha, gamma, u0],
        [0.0,   beta,  v0],
        [0.0,   0.0,   1.0]
    ])

    return _K, B


def determine_angle(x1: list, x2: list, w: ndarray):

    x1 = np.array(x1 + [1])
    x2 = np.array(x2 + [1])

    cosA = (x1 @ w @ x2) / np.sqrt((x1 @ w @ x1) * (x2 @ w @ x2))
    return np.degrees(np.arccos(cosA))


if __name__ == "__main__":

    frame : Frame = Frame("img.png", "frame",
                          slider=Slider("slider", {
                              "qualityLevel": [1, 100],     # (0, 1]
                              "minDistance": [10, 100],     # > 0
                              "blockSize": [3, 15]          # >= 3
                          })).toGray()
    frame.slider.last_params = {"qualityLevel" : .01, "minDistance" : 10, "blockSize" : 3}
    print(frame.slider.last_params)

    data = {
        Frame.CALLBACK_AMOUNT: [12],
        Frame.CALLBACK_POINTS: []
    }
    frame.set_mouse_callback(**data)    # initiate callback
    determine_good_points(frame, data)

    frame.slider()                      # initiate slider

    isStage2 : bool = False
    isStage3 : bool = False
    selected_points : list = []
    selected_points_sorted : list = []

    while True:

        determine_good_points(frame, data, flag=1)
        key = frame.copy().draw_point(data[Frame.CALLBACK_POINTS]).show(5)

        if ord('c') == key:    # press c when satisfied, to continue with sorting the points
            print("STAGE 2\n")
            isStage2 = True
            assert len(data[Frame.CALLBACK_POINTS]) == 12, "Not enough points to continue"
            selected_points = data[Frame.CALLBACK_POINTS].copy()

            data[Frame.CALLBACK_POINTS].clear()
            data[Frame.CALLBACK_AMOUNT][0] = 2

        elif ord('s') == key and len(selected_points_sorted) == len(selected_points) == 0:  # s to skip
            print("SKIPPING TO STAGE 3\n")
            assert len(data[Frame.CALLBACK_POINTS]) == 12, "Not enough points to continue"
            selected_points_sorted = data[Frame.CALLBACK_POINTS]
            isStage2 = False
            isStage3 = True

        elif ord('q') == key:   # q to quit
            frame.destroy()
            break


        if isStage2:
            # select outermost corners
            # only select 2 points
            # if first point selected, sort the first 4 points and remove them from the list

            if len(data[Frame.CALLBACK_POINTS]) == 1 and len(selected_points_sorted) == 0:

                _, distances = Frame.nearest_point(data[Frame.CALLBACK_POINTS][0], selected_points, flag=1)

                rect = list(map(lambda i: selected_points[i], distances[:4]))
                selected_points_sorted.extend(rect)     # add the 4 points to sorted list

                for each in rect:      # pop the 4 points
                    selected_points.remove(each)

                frame.copy().draw_point(selected_points_sorted).show(0)    # draw sorted and wait for user to select another point and a key press
                # for each in rect:                      # zu cheken ob richtig sortiert
                #     frame.copy().draw_point(each).show(0)

            if len(data[Frame.CALLBACK_POINTS]) == 2 and len(selected_points_sorted) == 4:

                _, distances = Frame.nearest_point(data[Frame.CALLBACK_POINTS][1], selected_points, flag=1)

                rect = list(map(lambda i: list(selected_points[i]), distances[:4]))
                selected_points_sorted.extend(rect)

                frame.copy().draw_point(selected_points_sorted).show(0)
                # for each in rect:                     # zu cheken ob richtig sortiert
                #     frame.copy().draw_point(each).show(0)

                for each in rect:
                    selected_points.remove(each)

                selected_points_sorted.extend(selected_points)
                isStage3 = True
                isStage2 = False

                print("Stage 2 DONE!")
                frame.copy().draw_point(selected_points_sorted).show(2000)
                # for each in selected_points:          # zu cheken ob richtig sortiert
                #     frame.copy().draw_point(each).show(0)


        elif isStage3:
            assert len(selected_points_sorted) == 12, "Not enough points to continue"
            print("\nSTAGE 3\n")
            # use sorted points to find H1, H2, H3
            H1 = determine_H(selected_points_sorted[:4])
            H2 = determine_H(selected_points_sorted[4: 8])
            H3 = determine_H(selected_points_sorted[8: ])

            K, w = determine_IAC_standard([H1, H2, H3])
            K /= K[-1, -1]
            print(f"K: \n{K.round(2)}\n")
            print(f"angle in degrees: {determine_angle(selected_points_sorted[0], selected_points_sorted[4], w)}")

            # print(determine_IAC([H1, H2, H3]))
            data[Frame.CALLBACK_AMOUNT][0] = 12
            data[Frame.CALLBACK_POINTS].clear()
            selected_points = selected_points_sorted = []

            isStage2 = False
            isStage3 = False

            determine_good_points(frame, data)
            frame.waitUserKey(0)



