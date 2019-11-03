import numpy as np
import cv2

fileim = "509401.jpg"
img = cv2.imread(fileim)

img_or = cv2.GaussianBlur(img, (5, 5), 0)

hsv = cv2.cvtColor(img_or, cv2.COLOR_BGR2HSV)


low_yellow = np.array([18, 94, 140])
up_yellow = np.array([48, 255, 255])

mask1 = cv2.inRange(hsv, low_yellow, up_yellow)


lower_white = np.array([84, 89, 56], dtype=np.uint8)
upper_white = np.array([178, 175, 104], dtype=np.uint8)


mask2 = cv2.inRange(hsv, lower_white, upper_white)


img = cv2.resize(mask1, (640,480))
img2 = cv2.resize(mask2, (640,480))
img_or= cv2.resize(img_or, (640,480))

cv2.imshow('res', img)
cv2.imshow('res2', img2)
cv2.imshow("rest3", img_or)
cv2.waitKey(0)
cv2.destroyAllWindows()
