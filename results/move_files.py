import glob
import os 

import shutil

def main():
    file_list = glob.glob("./*.jpg")
    print(len(file_list))

    for path in file_list:
        file_name = path.split(".jpg")[0]
        #print(file_name)
        name_label = file_name.split("-")
        if len(name_label) == 2:
            #label = name_label[0]
            label =name_label[-1]
            my_image_name_path = str(label[0]) + ".jpg"
            #print(my_image_name_path)
            file_name = name_label[0].split("./")[-1]
            os.makedirs(f"{label}",exist_ok=True)
            src = path
            print(src)
            dst_path = os.path.join(f"{label}", file_name + ".jpg")
            print(dst_path)
            shutil.move(src, dst_path)


if __name__ == "__main__":
    main()
