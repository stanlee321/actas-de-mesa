import numpy as np
import cv2

fileim = "509401.jpg"
img = cv2.imread(fileim)

scale = 4
h,w,_ = img.shape

print(h,w)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

##print(f"CONTOURS: {contours}")

print(len(contours))
new_cont = []

for c in contours:
    if len(c) > 60:
        #x_mean = map(lambda v : v[0], c)
        print(c.shape)
        # print(c[2].shape)
        # print(c[2])
        # for c_i in c:
        #     x_mean = np.sum(c_i[0])
        # new_cont.append(c)

print(len(new_cont))

frame = cv2.drawContours(img, new_cont, -1,(0,0,255),3)
frame = cv2.resize(frame, (int(w/scale), int(h/scale)))


cv2.imshow("rest3", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
