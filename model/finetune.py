from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import cv2
import numpy as np


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)


class MyModel:
    def __init__(self):


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = MnistResNet().to(self.device)

        try:
            self.model.load_state_dict( torch.load("./model/mnist_model_46_36.pth", map_location="cpu"))
        except:
            self.model.load_state_dict( torch.load("mnist_model_46_36.pth", map_location="cpu"))

        self.model.eval()

        self.mnist = MNIST(download=True, train=True, root=".").train_data.float()
        self.test_transforms = Compose([ Resize(((46, 46))), ToTensor(), Normalize((self.mnist.mean()/255,), (self.mnist.std()/255,))])

        self.BINARY_THREHOLD = 180

    def finetune(self):

        for param in self.model.parameters():
            param.require_grad = False

        print(self.model)

    def predict(self, image):
            
        with torch.no_grad():
            self.model.eval()
            image_tensor = self.test_transforms(image).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input_ = Variable(image_tensor)
            input_ = input_.to(self.device)
            output = self.model(input_)
            index = output.data.cpu().numpy().argmax()
            return index


    def image_smoothening(self, img):
        ret1, th1 = cv2.threshold(img,self.BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    def remove_noise_and_smooth(self, file_name):
        img = cv2.imread(file_name, 0)
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


    def preprocess_image(self, image):

        image = cv2.GaussianBlur(image, (7, 7), 0)

        h,w,c = image.shape
        
        image[image <128] = 0

        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_val = np.array([110,25,50])
        upper_val = np.array([255,255,255])
        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_val, upper_val)

        # Bitwise-AND mask and original image
        image = cv2.bitwise_and(image,image, mask= mask)
        
        # invert the mask to get black letters on white background
        image = cv2.bitwise_not(image)

        #image = image[int(w*0.2): int(w*0.9),int(h*0.2): int(h*0.8)] 

        #rimage = cv2.resize(image,(28,28))
        #rimage= np.moveaxis(rimage, 2, 0)
        #resized_image = np.reshape(rimage,(28,28,1))

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # rimage = np.expand_dims(im_gray, axis=0)
        # rimage= np.rollaxis(rimage, 3, 1) 
        # 


        return im_gray


    def main(self):

        file_in = "211851-1-2 copy.jpg"

        #image = cv2.imread(file_in)

        image=self.remove_noise_and_smooth(file_in)

        #image = preprocess_image(image)

        cv2.imwrite("test.jpg", image)
        
        to_pil = transforms.ToPILImage()

        pil_image = to_pil(image)
        index = predict(pil_image)

        print(index)

    def main_prediction(self, image_np):
        image_np = self.remove_noise_and_smooth_numpy(image_np)
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image_np)
        index = self.predict(pil_image)
        return index

if __name__ == "__main__":
    model = MyModel()

    model.finetune()