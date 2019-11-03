from PIL import Image
import numpy as np
import os

from .libs.prepro import remove_color_and_add_filter, imcrop_tosquare
from matplotlib import cm
import cv2
from fastai.vision import * # fastai 1.0
from fastai import *
from torchvision.models import *


######## RUN THIS ON A NEW MACHINE ##########

class MyModel:

    def __init__(self):
        self.counter=0

        # load the pre-trained network
        print("[INFO] loading pre-trained network...")
        arch = resnet50                             # specify model architecture
        base_path = "./model/models_fastai/"
        #MODEL_PATH = str(arch).split()[1] + '_stage1'
        MODEL_PATH = str(arch).split()[1] + '_stage2'

        empty_data = ImageDataBunch.load_empty(base_path) #this will look for a file named export.pkl in the specified path
    
        self.learner = cnn_learner(empty_data, arch).load(MODEL_PATH)
        
        
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

    # This function will convert image to the prediction format
    def imageToTensorImage(self, numpy_img):
        bgr_img = numpy_img
        b,g,r = cv2.split(bgr_img)
        rgb_img = cv2.merge([r,g,b])
        # crop to center to the correct size and convert from 0-255 range to 0-1 range
        H,W,C = rgb_img.shape

        rgb_img = rgb_img / 255.0

        return vision.Image(px=pil2tensor(rgb_img, np.float32))

    def main_prediction(self, numpy_img):
        #print("INPUT IMAGE ", numpy_img.shape)
        img_i = imcrop_tosquare(numpy_img)

        img_or = remove_color_and_add_filter(img_i)
        
        #print("OUTPUT IMAGE: ",img_or.shape )
        img_or_r = cv2.resize(img_or, (28,28))

        img = self.imageToTensorImage(img_or_r)
        pred = self.learner.predict(img)[0]


        pred = int(float(str(pred)))

        # #print(pred)

        path = f"./results/{self.counter}-{pred}-i.jpg"
        # path_2 = f"./results/{self.counter}-{pred}.jpg"

        cv2.imwrite(path, img_i)
        
        # cv2.imwrite(path_2, img_or)

        self.counter +=1
        return pred

