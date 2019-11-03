import matplotlib.pyplot as plt
import glob
import cv2
import os
import shutil


category=[]
plt.ion()

f= open("labels.txt","w+")

def write_row(path, label):
    f.write(f"{path},{label} \n")



def main():
    images_list = glob.glob("cuts/partidos/numbers/i_letter/*.jpg")
    size =len(images_list)
    print("size:", size)

    for i, im_p in enumerate(images_list):
        img = cv2.imread(im_p)
        plt.imshow(img)
        plt.pause(0.05)
        cat = input('category: ')
        file_n = im_p.split("/")[-1]
        print(f"{i}/{size}")
        shutil.move(im_p, f"labels/{cat}/{file_n}")

        #write_row(im_p, category[-1] )


main()