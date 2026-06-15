# Frame

`Frame` is a `np.ndarray` subclass that integrates OpenCV operations directly on the array instance.

**Core design goal: one object, all transformations — no intermediate copies, no `dst` variables.**

## The Problem

```python
# standard cv2 / numpy
import cv2 as cv

frame = cv.imread("img.png")
gray    = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
resized = cv.resize(blurred, dsize=(640, 480))
cv.imshow("preview", resized)
cv.waitKey(0)
```

```python
# Frame
Frame("img.png", "preview").toGray()(cv.GaussianBlur, ksize=(5, 5), sigmaX=0).rescale(0.5).show(0)
# or 
Frame("img.png", "preview")(cv.cvtColor, code=cv.COLOR_BGR2GRAY)(cv.GaussianBlur, ksize=(5, 5), sigmaX=0)(cv.resize, dsize=(640, 480)).show(0)
```

The same Python object persists through every transformation.
Shape changes are handled by rewriting `ndarray.dimensions` and `ndarray.strides`
directly via `ctypes` — no reallocation, no new instance.

## Usage

```python
from Frame import Frame
from TrackBar import Slider

# load → chain transforms → display
Frame("img.png", "preview").toGray().rescale(0.5).show(0)

# with trackbar
frame = Frame("img.png", "preview", 
              slider=Slider("controls", {
                  "threshold": [127, 255],
                  "ksize":     [3,   20]
              })
)
frame.slider() # to initiate trackbar
```

### Mouse callbacks

```python
data = {
    Frame.CALLBACK_AMOUNT: [3],
    Frame.CALLBACK_POINTS: []
}
frame.set_mouse_callback(**data)
# double-click  → place point
# shift + double-click → remove nearest point (within 20px)
```

## Examples

### Epipolar Geometry Visualizer
[`examples/epipolarlines.py`](examples/epipolarlines.py)

Loads two images from a dataset, computes the Fundamental Matrix from known projection matrices,
and lets the user place points interactively — epipolar lines update live in the other frame.

```python
frames, camera_matrices, K = load_images(mats=[0, 14])
frames = [frame.rescale(0.5) for frame in frames]

E, R = determine_E_using_P(camera_matrices, K)
F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

frames[0].set_mouse_callback(**data_frame1)
frames[1].set_mouse_callback(**data_frame2)
```

Controls:
- `double-click` — place point → epipolar line appears in the other frame
- `shift + double-click` — remove nearest point
- `d` — clear all points
- `q` — quit

---

### Online Camera Calibration using IAC (Zhang's Method)
[`examples/online_calib_with_IAC.py`](examples/online_calib_with_IAC.py)

Detects corners via `goodFeaturesToTrack` (tunable via trackbar), walks through a 3-stage UI
to sort and assign points to checkerboard squares, then computes the intrinsic matrix **K**
via the Image of the Absolute Conic (DLT + SVD).

```python
frame = Frame("img.png", "frame",
              slider=Slider("slider", {
                  "qualityLevel": [1, 100],
                  "minDistance":  [10, 100],
                  "blockSize":    [3, 15]
              })).toGray()
```

Stages:
- **Stage 1** — adjust trackbar to tune corner detection, manually add/remove points, press `c`
- **Stage 2** — click 2 outer corners per square to sort the 12 points into correct order
- **Stage 3** — K and the angle between two projection-center rays are printed

Output:
```
K:
[[ 832.14    0.     312.56]
 [   0.    831.89  241.03]
 [   0.      0.      1.  ]]

angle in degrees: 43.27
```

## Project Structure

```
├── examples/
│   ├── epipolarlines.py
│   └── online_calib_with_IAC.py
├── Capture.py
├── Datamodel.py
├── Flow.py            # (planned) stage-based pipeline
├── Frame.py           # core — ndarray subclass, ctypes in-place mutation, mouse callbacks
├── Model.py
├── TrackBar.py        # Slider — cv.createTrackbar wrapper
└── Video.py
```

## How `__call__` works

`frame(func, **kwargs)` applies `func(self, ...)` and then:

1. rewrites `ndarray.nd`, `.dimensions`, `.strides` via ctypes if shape changed
2. copies the result into `self[:]` in-place
3. deletes the intermediate array

`__getattribute__` intercepts all inherited ndarray methods (non-dunder, not in `Frame.__dict__`)
and routes them through `__call__`, so `frame.transpose()`, `frame.astype(np.float32)` etc.
all mutate and return `self`.

## Requirements

- Python 3.10+
- NumPy
- OpenCV (`cv2`)

## Status

Side project — developed alongside university CV coursework.
Planned: PyTorch interop, `Flow` pipeline, serialization, frequency domain support.