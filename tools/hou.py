

# I am trying to detect lines in parking as shown below

# Empty parking lot

# What I hope to get is the clear lines and (x,y) position in the crossed line, however the result is not very promising

# Parking lot with Hough Lines drawn

# I guess it is due to two main reasons

#     some lines are very broken or missing even human eyes can clearly identify them. (Even HoughLine can help to connect some missing lines since HoughLine sometimes would connect unnecessary lines together, so I 'd rather to do it manually)

#     there are some repeated lines

# The general pipeline for the work is shown as below
# 1. select the some specific colors (white or yellow)

import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt


filein="509401.jpg"
img = cv2.imread(filein)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:

    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Draw the lines on the  image
#lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

img = cv2.resize(img, (640,480))

cv2.imshow('res2', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
