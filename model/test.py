import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import cv2
from collections import OrderedDict

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MyModel:
    def __init__(self):
                
        #model = Net()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net()
        #model.load_state_dict( torch.load("mnist-b07bb66b.pth", map_location="cpu"))
        self.model.load_state_dict(torch.load('./model/mnist_cnn.pt'))
        self.model.eval()


        self.test_transforms = transforms.Compose([transforms.Resize(28),
                                            transforms.ToTensor(),
                                            ])
        self.BINARY_THREHOLD = 180


    def predict(self, image):
            
        with torch.no_grad():
            self.model.eval()

            image_tensor = self.test_transforms(image).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input_ = Variable(image_tensor)
            input_ = input_.to(self.device)
            output = self.model(input_)
            #print(f"output211851-1-2 {output}")
            index = output.data.cpu().numpy().argmax()
            return index


    def image_smoothening(self, img):
        ret1, th1 = cv2.threshold(img, self.BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
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
        h,w,c = image.shape
        image[image <128] = 0

        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_val = np.array([110,30,50])
        upper_val = np.array([255,255,255])
        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_val, upper_val)

        # Bitwise-AND mask and original image
        image = cv2.bitwise_and(image,image, mask= mask)
        
        # invert the mask to get black letters on white background
        image = cv2.bitwise_not(image)


        #image = image[int(w*0.2): int(w*0.9),int(h*0.2): int(h*0.8)] 

        rimage = cv2.resize(image,(28,28))
        #rimage= np.moveaxis(rimage, 2, 0)
        #resized_image = np.reshape(rimage,(28,28,1))

        im_gray = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)
        #im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        # rimage = np.expand_dims(im_gray, axis=0)
        # rimage= np.rollaxis(rimage, 3, 1) 
        # 

        return im_gray

    def main(self):

        file_in = "211851-1-2.jpg"
        image = self.remove_noise_and_smooth(file_in)

        #image = cv2.imread(file_in)
        #image = preprocess_image(image)

        cv2.imwrite("test.jpg", image)
        to_pil = transforms.ToPILImage()

        pil_image = to_pil(image)
        index = self.predict(pil_image)

        print(index)

    def main_prediction(self, image_np):
        image_np = self.remove_noise_and_smooth_numpy(image_np)
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image_np)
        index = self.predict(pil_image)

        return index

if __name__ == "__main__":
    my_model = MyModel()

    my_model.main()