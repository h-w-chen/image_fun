#!/usr/bin/env python3

### to cartoonize live video from the default web cam

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

kernel = np.ones((3,3), np.uint8)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame, 1)  # mirror(horizental) flip

    # convert to sketch
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted = 255 - gray

    # Blur the inverted image using a Gaussian filter
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    # Blend the grayscale image with the blurred inverted image using the "color dodge" blending mode
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    # make it thicker
    sketch = cv2.erode(sketch, kernel, iterations=1)
#    sketch = cv2.medianBlur(sketch, ksize=3)
    sketch = cv2.bilateralFilter(sketch, 13, 0, 0)

    cv2.imshow("live", sketch)

    c = cv2.waitKey(1)
    if c == 27: # Esc
        break

cap.release()
cv2.destroyAllWindows()
