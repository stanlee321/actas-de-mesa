from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from PIL import Image
import numpy as np
import os

from keras.models import load_model
from .libs.prepro import remove_color_and_add_filter, imcrop_tosquare

from matplotlib import cm
import cv2


class MyModel:

    def __init__(self):
        MODEL1 = "mnistCNN-new-simple.h5"
        MODEL2 = "mnistCNN-new-simple-two.h5"
        MODEL3 = "mnistCNN-new-simple-tree.h5"
        MODEL4 = "mnistCNN-new-simple-four.h5"
        MODEL5 = "mnistCNN-new-simple-five.h5"
        MODEL6 = "mnistCNN-new-simple-six.h5"
        MODEL7 = "mnistCNN-new-simple-seven.h5"
        self.counter=0
        # load the pre-trained network
        print("[INFO] loading pre-trained network...")
        try:
            self.model = load_model(f"{MODEL7}")
        except:
            self.model = load_model(f"./model/{MODEL7}")
        
        self.BINARY_THREHOLD = 180


    def image_smoothening(self, img):
        ret1, th1 = cv2.threshold(img,self.BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    def remove_noise_and_smooth_numpy(self, image_np):
        img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        h,w = img.shape

        #img[img <128] = 0
        img = img[int(w*0.2): int(w*0.9),int(h*0.2): int(h*0.9)] 

        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = self.image_smoothening(img)
        or_image = cv2.bitwise_or(img, closing)
        return or_image


    def main_prediction(self, numpy_img):
        #img = Image.fromarray(np.uint8(cm.gist_earth(numpy_img)*255)).convert("L")
        #image_np = self.remove_noise_and_smooth_numpy(numpy_img)

        #img = Image.open('211851-0-2.jpg')
        #img = np.resize(numpy_img, (28,28,1))
        #im2arr = np.array(img)
        # print("input shape")
        # print(numpy_img.shape)
        #img = np.resize(img, (28,28,1))
        img = imcrop_tosquare(numpy_img)

        # print("new image crip shape")
        # print(img.shape)
        img = remove_color_and_add_filter(img)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        img = cv2.resize(img, (28,28))

        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)

        # make predictions on the input image
        pred = self.model.predict(im2arr)
        pred = pred.argmax(axis=1)[0]
        
        #cv2.imwrite(f"./results/{self.counter}_{pred}.jpg", img)
        self.counter +=1
        return pred

