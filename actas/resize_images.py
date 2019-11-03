import cv2
import glob
import os


images_list = glob.glob("mesas/*.jpg")

data_size = len(images_list)

f= open("bad_images.txt","w+")

def write_row(text):
    f.write(f"bad image: {text} \n")


for image_name in images_list:
    print(image_name)
    try:

        img = cv2.imread(image_name)
        h,w,c = img.shape

        new_w = 420 # ~6
        new_h = 260 # ~6

        img_r = cv2.resize(img, (new_w, new_h))

        new_name = os.path.join("mesas_resized", image_name.split("/")[-1])

        cv2.imwrite(new_name, img_r)

    except Exception as e:
        print(e)
        write_row(image_name)
