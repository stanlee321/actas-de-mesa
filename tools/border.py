import numpy as np
import cv2
import scipy.ndimage as ndi

class BorderFinder:

    def __init__(self):
        ###
        pass


    def norm_image(self, image):
        """
        Norm image to constant shape
        """
        img = cv2.resize(image, (2500, 1600))
        return img

    def image_smoothening(self,img):
        BINARY_THREHOLD = 180

        ret1, th1 = cv2.threshold(img,BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    def remove_noise_and_smocoth_numpy(self,image_np):
        img = image_np.copy()

        img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        h,w = img.shape
        #img[img <128] = 0
        #img = img[int(w*0.2): int(w*0.9),int(h*0.2): int(h*0.9)] 

        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 
                                        255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 41)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = self.image_smoothening(img)
        or_image = cv2.bitwise_or(img, closing)
        
        return or_image

    def main(self, image_path):
        img = cv2.imread(image_path)
        img = self.norm_image(img)
        h,w,c = img.shape

        n_h = int(h/4)
        n_w = int(w/4)

        img = cv2.resize(img, (n_w, n_h ))

        gray = self.remove_noise_and_smocoth_numpy(img)
        
        img = gray.copy()
        """
        --------------------
        -    .p1       .p2 -
        -                  -
        -    .p3       .p4 -
        --------------------
        """

        p1 = [(0,0), (int(0.1*n_w), int(0.1*n_h))]

        c_p1_p1 = p1[0]
        c_p1_p2 = p1[1]

        p1_img = img[c_p1_p1[1]:c_p1_p2[1], c_p1_p1[0]:c_p1_p2[0]]

        print(p1_img.shape)
        cv2.imshow("Line edges_1", p1_img) 


        p2 = ((0, int(n_h*0.9)), (int(0.1*n_w), int(n_h)))

        c_p2_p1 = p2[0]
        c_p2_p2 = p2[1]
        
        p2_img = img[c_p2_p1[1]:c_p2_p2[1], c_p2_p1[0]:c_p2_p2[0]]

        print(p2_img.shape)
        cv2.imshow("Line edges_2", p2_img) 


        p3 = ((int(0.9*n_w), int(0.9*n_h)), (int(n_w), int(n_h)))

        c_p3_p1 = p3[0]
        c_p3_p2 = p3[1]
        
        p3_img = img[c_p3_p1[1]:c_p3_p2[1], c_p3_p1[0]:c_p3_p2[0]]

        print(p3_img.shape)
        cv2.imshow("Line edges_3", p3_img) 


        p4 = [(int(0.9*n_w),0), (int(n_w), int(0.1*n_h))]

        c_p4_p1 = p4[0]
        c_p4_p2 = p4[1]
        
        p4_img = img[c_p4_p1[1]:c_p4_p2[1], c_p4_p1[0]:c_p4_p2[0]]

        print(p4_img.shape)
        cv2.imshow("Line edges_4", p4_img) 

        # CV    
        #gray = cv2.cvtColor(p1_img, cv2.COLOR_BGR2GRAY)

        smooth = ndi.filters.median_filter(gray, size=2)


        edges = cv2.Canny(smooth,50,150,apertureSize = 3)


        # Defining a kernel length
        kernel_length = np.array(img).shape[1]//80
        
        # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))# A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        
        # Morphological operation to detect vertical lines from an image
        img_temp1 = cv2.erode(p3_img, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
        cv2.imshow("verticle_lines",verticle_lines_img)# Morphological operation to detect horizontal lines from an image
        
        img_temp2 = cv2.erode(p3_img, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
        cv2.imshow("horizontal_lines",horizontal_lines_img)


        # lines_2 = cv2.HoughLines(edges,1,np.pi/180, 120)

        # for line in lines_2:
        #     rho,theta = line[0]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a*rho
        #     y0 = b*rho
        #     x1 = int(x0 + 1000*(-b))
        #     y1 = int(y0 + 1000*(a))
        #     x2 = int(x0 - 1000*(-b))
        #     y2 = int(y0 - 1000*(a))
        #     cv2.line(p1_img,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow("Line Detection", p1_img) 
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    border = BorderFinder()
    path = "../509401.jpg"
    border.main(path)