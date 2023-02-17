#!/usr/bin/env python3

### to warp an image as both vertical & horizental waves

import cv2
import numpy as np
import math

img = cv2.imread("/home/howell/work/t-cv/01_Getting_Started_with_Images/car1.jpg", cv2.IMREAD_COLOR)
rows, cols = img.shape[:2]

img_out = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(20.0 * math.sin(2*3.14*i/150))
        offset_y = int(20.0 * math.cos(2*3.14*j/150))
        if i + offset_y < rows and j + offset_x < cols:
            img_out[i,j] = img[(i+offset_y)%rows, (j + offset_x) %cols]
        else:
            img_out[i,j] = 0

cv2.imshow('omg', img_out)
cv2.waitKey()
