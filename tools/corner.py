import numpy as np
import cv2
from matplotlib import pyplot as plt


fileim = "509401.jpg"
img = cv2.imread(fileim)
img = cv2.GaussianBlur(img, (5, 5), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)


img = cv2.resize(img, (640,480))

cv2.imshow('res2', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
