#!/usr/bin/env python3

# excertp of seam carving, from book OpenCV 3.x with Python by Examples

import sys
import cv2
import numpy as np


def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    # retur the weighted summation 0.5*X + 0.5*Y
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0,5, 0)


def findVerticalSeam(img, energy):
    rows, cols = img.shape[:2]
    seam = np.zeros(img.shape[0])
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0, :]=np.zeros(img.shape[1])
    edge_to = np.zeros(img.shape[:2])
    for row in range(rows - 1):
        for col in range(cols -1):
            if col != 0 and dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]:
                dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1]
                edge_to[row+1, col - 1] = 1
            if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]:
                dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col]
                edge_to[row+1, col - 1] = 0
            if col != cols -1 and dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]:
                dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1]
                edge_to[row+1, col+1] = -1
    seam[rows-1] = np.argmin(dist_to[rows-1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i-1] = seam[i] + edge_to[i, int(seam[i])]
    return seam

def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(int(seam[row]), cols-1):
            img[row, col] = img[row, col+1]
    img = img[:, 0:cols-1]
    return img

def add_vertical_seam(img, seam, iters):
    seam = seam + iters
    rows, cols = img.shape[:2]
    zero_col_mat = np.zeros((rows,1,3), dtype=np.uint8)
    img_extended = np.hstack((img, zero_col_mat))
    for row in range(rows):
        for col in range(cols, int(seam[row], -1)):
            img_extended[row,col] = img[row, col-1]
            for i in range(3):
                v1 = img_extended[row, int(seam[row])-1, i]
                v2 = img_extended[row, int(seam[row])+1, i]
                img_extended[row, int(seam[row]), i] = (int(v1) + int(v2))/2

    return img_extended

def draw_rect(evt, x, y, flags, params):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt, img_orig
    if evt == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
    elif evt == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            top_left_pt, bottom_right_pt = (x_init, y_init), (x,y)
            img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]
            cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
    elif evt == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left_pt, bottom_right_pt = (x_int, y_init), (x,y)
        img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]
        cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        rect_final = (x_init, y_init, x-x_init, y-y_init)
        remove_obj(img_orig, rect_final)

def compute_energy_matrix_modified(img, rect_roi):
    e_matrix = compute_energy_matrix(img)
    x,y,w,h = rect_roi
    e_matrix[y:y+h, x:x+w] = 0
    return e_matrix

def remove_obj(img, rect_roi):
    num_seams = rect_roi[2] + 10
    energy = compute_energy_matrix_modified(img, rect_roi)
    # loop to rm one seam one time
    for i in range(num_seams):
        seam = find_vertical_seam(img, seam)
        img = remove_vertical_seam(img, seam)
        x,y,w,h=rect_roi
        # recal energy matrix after removing the seam
        energy = comp_energy_matrix_modified(img, (x,y,w-i,h))
    img_output = np.copy(img)
    # loop to fill up
    for i in range(num_seams):
        seam = find_vertical_seam(img, energy)
        img = remove_vertical_seam(img, seam)
        img_output = add_vertical_seam(img_output, seam, i)
        energy = compute_energy_matrix(img)

    cv2.imshow('input', img_input)
    cv2.imshow('output', img_output)
    cv2.waitKey()


if __name__ == '__main__':
    # Load the image
    img_input = cv2.imread('/home/howell/work/t-cv/01_Getting_Started_with_Images/car1.jpg')
    drawing = False
    img = np.copy(img_input)
    img_orig = np.copy(img_input)

    cv2.namedWindow('qqq')
    cv2.setMouseCallback('qqq', draw_rect)
    while True:
        cv2.imshow('qqq', img)
        c = cv2.waitKey(10)
        if c == 27:
            break

    cv2.destroyAllWindows()
