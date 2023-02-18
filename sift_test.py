#!/usr/bin/env python3

import cv2

img = cv2.imread("/home/howell/work/t-cv/01_Getting_Started_with_Images/car1.png", cv2.IMREAD_COLOR)

#sift = cv2.xfeatures2d.SIFT_Create()
sift = cv2.xfeatures2d.SIFT_create()
#keypoints = sift.detect(img, None)
keypoints, descriptors = sift.detectAndCompute(img, None)
#print(descriptors)

cv2.drawKeypoints(img, keypoints, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift', img)

cv2.waitKey()
cv2.destroyAllWindows()
