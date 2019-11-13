import cv2
import glob
import os
import pickle
from tqdm import tqdm
#from model.test import MyModel
#rom model.restnet import MyModel
from model.fastai import MyModel
import argparse

# sorter
from operator import itemgetter
from pascal_voc_writer import Writer


class Partido:
    def __init__(self, name="", box_name=[], box_count=[]):
        self.name = name
        self.box_name = box_name
        self.box_count = box_count

class ValidosAndOthers:
    def __init__(self, name="", box_name=[], box_count=[]):
            self.name = name
            self.box_name = box_name
            self.box_count = box_count



class ImageHanlder:
    def __init__(self, cut_numbers=True, draw_results=False, all_boxes=True, draw_rects=False, fixed=False):
        self.font                   = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale              = 0.5
        self.fontColor              = (255,0,255)
        self.lineType               = 2

        # If all 24 boxes was discovered with the CV alog, map his ID to  his real-name
        self.name_map               = {"22": "CC", "14": "MAS"}

        self.cut_numbers=cut_numbers
        # Instantiate mnist model
        self.model = MyModel()

        # Bounding rectangles placholder
        self.outputs = []

        # For handle the data 
        self.data_handler = []

        # Placholder for the partidos.
        self.partidos_todos = []
        self.partidos = []

        # If draw results
        self.draw_results = draw_results

        # If you only want to filter CC and MAS
        self.all_boxes = all_boxes
        
        # Expected position of partidos in 1088, 872 Resolution image
        # in the acta region.

        # CC Name box
        # P1 : 14,40
        # P2 : 520,90
        
        # Region containing the information about the Mesas to count
        self.P1 = (90*4, 98*4)
        self.P2 = (272*4, 218*4)
        
        
        self.draw_rects = draw_rects
        
        partidos_d = [24, 22, 20, 18, 16, 14, 12, 10, 8, 6]
        partidos = [23, 21, 19, 17, 15, 13, 11, 9, 7]

        otros_d = [4, 2, 6]
        otros = [5, 3, 1]

        self.partidos_todos = partidos + partidos_d

        self.otros_todos = otros + otros_d

        self.ids = partidos + partidos_d   + otros + otros_d
        
        self.base_path = "results/"

        self.fixed = fixed

        self.normal_shape = (2500, 1600)


        self.name_maping = {
            1: "Votos Nulos Presidente",
            2: "Votos Nulos Diputado",
            3: "Votos Blancos Presidente",
            4: "Votos Blancos Diputado",
            5: "Votos Validos Presidente",
            6: "Votos Validos Diputado",
            7: "PAN-BOL Presidente",
            8: "PAN-BOL Diputado",
            9: "MNR Presidente",
            10: "MNR Diputado",
            11: "PDC Presidente",
            12: "PDC Diputado",
            13: "21F Presidente",
            14: "21F Diputado ",
            15: "MAS IPSP Presidente",
            16: "MAS IPSP Diputado",
            17: "UCS Presidente" ,
            18: "UCS Diputado",
            19: "MTS Presidente",
            20: "MTS Diputado",
            21: "FPV Presidente",
            22: "FPV Diputado",
            23: "CC Presidente",
            24: "CC Diputado"
        }

        self.name_maping_results = {
            25: "Votos Nulos Presidente, Result",
            26: "Votos Nulos Diputado, Result",
            27: "Votos Blancos Presidente, Result",
            28: "Votos Blancos Diputado, Result",
            29: "Votos Validos Presidente, Result",
            30: "Votos Validos Diputado, Result",
            31: "PAN-BOL Presidente, Result",
            32: "PAN-BOL Diputado, Result",
            33: "MNR Presidente, Result",
            34: "MNR Diputado, Result",
            35: "PDC Presidente, Result",
            36: "PDC Diputado, Result",
            37: "21F Presidente, Result",
            38: "21F Diputado, Result",
            39: "MAS IPSP Presidente, Result",
            40: "MAS IPSP Diputado, Result",
            41: "UCS Presidente, Result" ,
            42: "UCS Diputado, Result",
            43: "MTS Presidente, Result",
            44: "MTS Diputado, Result",
            45: "FPV Presidente, Result",
            46: "FPV Diputado, Result",
            47: "CC Presidente, Result",
            48: "CC Diputado, Result"
        }


    def combine_lists(self):
          # combine all the info
        #self.partidos_todos = self.partidos.extend(self.partidos_d)
        self.todos_los_fields = self.partidos + self.partidos_d + self.otros + self.otros_d

        #print(len(self.todos_los_fields))

    def write_row_debug_log(self, file_name, path_img, label):
        """
        Write log
        """
        os.makedirs(self.base_path, exist_ok=True)

        path = f"{self.base_path}/{file_name}.txt"

        with open(path,"a+") as f:
            f.write(f"{path_img},{label} \n")


    def write_row_results_log(self, file_name, image_name, counts, id):
        """
        Write the last results
        """
        os.makedirs(self.base_path, exist_ok=True)
        path = f"{self.base_path}/{file_name}.txt"

        with open(path,"a+") as f:
            f.write(f"{image_name},{counts},{id} \n")


    def norm_image(self, image):
        """
        Norm image to constant shape
        """
        img = cv2.resize(image, self.normal_shape)
        return img

    def cut_image(self, image, p1, p2, simple=False):
        """
        Crop any imagen given P1 and P2
        """
        x = p1[0]
        y = p1[1]

        w = p2[0]
        h = p2[1]
        
        if simple:
            crop_img = image[y:h, x:w]
        crop_img = image[y:y+h, x:x+w]
        return crop_img


    def scale_image(self, image):
        """
        Scale Down any image by a factor of 4
        """
        h,w,c=image.shape

        img_r = cv2.resize(image, (int(w/4), int(h/4)))
        
        # img = cv2.imshow('res2', img_r)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def find_contour_base(self, image, kernel):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        
        #--- performing Otsu threshold ---
        ret,thresh1 = cv2.threshold(gray, 0, 255, 
                                    cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
        #--- choosing the right kernel
        #--- kernel size of 3 rows (to join dots above letters 'i' and 'j')
        #--- and 10 columns to join neighboring letters in words and neighboring words
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        #---Finding contours ---
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        im2 = image.copy()

        return im2, contours

    def find_contour(self, image, image_path):
        #(30, 2) for letters
        im2, contours = self.find_contour_base(image, (7,2))

        boxes = []
        outputs =[]

        # Iterate over all contours
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            p1 = (x, y)
            p2 = (x + w, y + h)

            deltax = p2[0]-p1[0]
            deltay = p2[1]-p1[1]

            # in range? of a box?
            if (200<=(deltax)<=360) and (30<=(deltay)<=70):
                # print(deltax)
                # print(deltay)
                p1 = (p1[0], p1[1])
                p2 = (p2[0], p2[1])

                boxes.append([p1,p2])
                
                if (self.draw_rects):
                    cv2.rectangle(im2, p1, p2, (0, 255, 0), 2)

        def sortSecond(val): 
            return val[0]

        size = len(boxes)
        #print(f"TOTAL BOXES {size}")
        #boxes.sort(key=sortSecond)
        #print(boxes)
        # If all the rectanguler boxes was found.
        if size == 24:
            # print("size",size)

            # Iterate over the rectangular boxes
            for i, b in enumerate(boxes):
                index_or = i

                out_index = index_or + 1
                partido_id = str(out_index)

                # point  = b[0]

                # h,w,c = image.shape
                # half_x = w / 2

                # if (int(partido_id) in [2,4,6,8,10,12,14,16,18,20, 22, 24]) and ( point[0] < half_x):

                #     out_index -= 1

                # if (int(partido_id) in [1,3,5,7,9,11,13,15,17,19, 21, 23]) and ( point[0] > half_x):
                #     out_index += 1

                outputs.append({out_index: b})
        else:
            """
            If detection is < 24 log this file
            """
            file_name = "detect_boxes_error_log"
            self.write_row_debug_log(file_name, image_path, size)

        return im2, outputs


    def find_contours_leters(self, image):

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 0, 155, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        return rects
    
    def calculate_area(self, p1,p2):
        w = p2[0] - p1[0]
        h = p2[1] - p1[1]
        area = w*h
        #print(area)
        return area
    
    def draw_rectangle_numpy(self, img, p1,p2):
        img = cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        return img

    def main(self, images_list):
        
        for i_path in tqdm(images_list, ascii=True, desc="Reading..."):
            ##print(f"IMAGE : {i_path}")
            try:
                # Try to open the image
                #i = "actas/mesas/200081.jpg"
                img = cv2.imread(i_path)

                if self.fixed:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_r   =   self.norm_image(img)
                # extract only the votation box
                c_image =   self.cut_image(img_r, self.P1, self.P2)

                # Obtain the full filename
                filename = i_path.split("/")[-1]

                # Obtain only the name without extension 
                filename_two = filename.split(".")[0]
                
                # Find the contours box rectangle for each partido 
                cont_img, self.outputs = self.find_contour(c_image, i_path)
                
                _writer = Writer(i_path, self.normal_shape[1], self.normal_shape[0])

                for o in self.outputs:
                    # Iterate over the rectangular boxes
                    for k, v in o.items():
                        #print(f"WORKING ON {k}")
                        partido_key_id = k  
                        
                        p1, p2 = v


                        if k in self.partidos_todos:
                            # Mueva Position tira 
                            n_p1 = (p1[0] + 340, p1[1])
                            n_p2 = (p2[0] + 180, p2[1])

                        if k in self.otros_todos:

                            # Mueva Position tira 
                            n_p1 = (p1[0] + 310, p1[1])
                            n_p2 = (p2[0] + 215, p2[1])

                        

                        # ADD Result boxes
                        xmin = p1[0] / self.normal_shape[0]
                        ymin = p1[1] / self.normal_shape[1]

                        xmax = p2[0] / self.normal_shape[0]
                        ymax = p2[1] / self.normal_shape[1]

                        xmin_n = n_p1[0] / self.normal_shape[0]
                        ymin_n = n_p1[1] / self.normal_shape[1]

                        xmax_n = n_p2[0] / self.normal_shape[0]
                        ymax_n = n_p2[1] / self.normal_shape[1]


                        # ::addObject(name, xmin, ymin, xmax, ymax)
                        label = self.name_maping[int(partido_key_id)]
                        label_n = self.name_maping_results[int(partido_key_id+24)]
                        


                        _writer.addObject(label, xmin, ymin, 
                                                xmax, ymax)

                        
                        _writer.addObject(label_n, xmin_n, ymin_n, 
                                                xmax_n, ymax_n)


                        """
                        p1 = ( int(xmin*self.normal_shape[0]),
                                int(ymin * self.normal_shape[1]))

                        p2 = (int(xmax *self.normal_shape[0] ), 
                                int(ymax*self.normal_shape[1]))


                        n_p1 = (int(xmin_n *self.normal_shape[0] ), 
                                int(ymin_n*self.normal_shape[1]))

                        n_p2 = (int(xmax_n* self.normal_shape[0]),
                                int(ymax_n*self.normal_shape[1]))


                        
                        
                        self.draw_rectangle_numpy(cont_img, p1, p2 )

                        self.draw_rectangle_numpy(cont_img, n_p1, n_p2 )


                        cv2.putText(cont_img, label, tuple(p1), self.font, 
                                                        self.fontScale,
                                                        self.fontColor,
                                                        self.lineType)

                        cv2.putText(cont_img, label_n, tuple(n_p1), self.font, 
                                                        self.fontScale,
                                                        self.fontColor,
                                                        self.lineType)

                        """
                # Write the acta with the number of results drawed on it.
                base_path = "actas/cuts/"

                os.makedirs(base_path, exist_ok=True)
                path = f"{base_path}{filename}"
                #print(f"SAVING IMAGE PRO en {path}")
                path_label = f"{base_path}{filename_two}_label.xml"

                _writer.save(path_label)
                cv2.imwrite(path, cont_img)
                
            except Exception as e:
                filename = i_path.split("/")[-1]
                _filename_error = "errorInOpenFile"
                self.write_row_debug_log(_filename_error, filename, e)
                print(e)
            #break

def load_fixed_data(path):
    with open(f'{path}', 'rb') as filehandle:
        # store the data as binary data stream
        fixed_list = pickle.load(filehandle)
    return fixed_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Actas de mesa to counts')

    parser.add_argument('--data_path', type=str, default="actas/images/", help='Where is your jpg image data. absolute path')
    parser.add_argument('--draw_results', type=bool,default=True, help='Input dir for videos')
    parser.add_argument('--all_boxes', type=bool, default=True, help='Output dir for image')
    
    parser.add_argument("--fixes", type=str, default="", help="If you are augmenting the fixed data list" )
    
    args = parser.parse_args()

    # Local aux variable for handle if is fixed data
    fixed = False

    images_path = args.data_path
    draw_results = args.draw_results
    all_boxes= args.all_boxes
    
    fixed_list_path = args.fixes

    # Images lists...
    images_list = glob.glob(f"{images_path}*.jpg")

    image_hanlder = ImageHanlder(cut_numbers=False, # For custom mnist dataset creation
                                draw_results=True,  # Draw the actas with the result drawed on it
                                all_boxes=all_boxes,     # Just find CC and MAS or all
                                draw_rects=False,
                                fixed=fixed
                                ) 

    image_hanlder.main(
        images_list
    )
