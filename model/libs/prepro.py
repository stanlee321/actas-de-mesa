# AUX FUNCTIONS
import math
import cv2
import numpy as np

from scipy import ndimage
# Some Preprocess
BINARY_THREHOLD = 180

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img,BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return blur

def remove_noise_and_smocoth_numpy(image_np):
    img = image_np.copy()

    img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    h,w = img.shape
    #img[img <128] = 0
    img = img[int(w*0.2): int(w*0.9),int(h*0.2): int(h*0.9)] 

    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 
                                      255, 
                                      cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,
                                    11, 4)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    
    return or_image

def remove_red_color_numpy(image_np):
    
    img = image_np.copy()
    
    lower = np.array((90 - 120, 30, 50))  #-- Lower range --
    upper = np.array((90 + 120, 255, 255))  #-- Upper range 
    
    mask = cv2.inRange(img, lower, upper)
    
    res = cv2.bitwise_and(img, img, mask= mask)  #-- Contains pixels having the gray color--

    res = cv2.bitwise_not(res)
    
   
    return res

def convert_to_gray(image_np):
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    return image_gray

  
def getBestShift(img):
  cy,cx = ndimage.measurements.center_of_mass(img)

  rows,cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
  
  
def remove_noise(img_or):
  img_or = remove_noise_and_smocoth_numpy(img_or)

  #print("img_or shape", img_or.shape)
  img_or = cv2.resize(img_or,(60,60))
  #print(img.shape)
  o_w, o_h = img_or.shape
  #print(img.shape)
  
  img_or = 255.0 - img_or

  return img_or

def remove_color_and_add_filter(numpy_i):
  img_or = remove_red_color_numpy(numpy_i)
  img = remove_noise(img_or)
  """
  try:
    while np.sum(img[0]) == 0:
      img = img[1:]
  except Exception as e:
    print(f"error_1 {e}")
    return img_or
  try:
    while np.sum(img[:,0]) == 0:
      img = np.delete(img,0,1)
  except Exception as e:
    print(f"error_2 {e}")
    return img_or
  try:
    while np.sum(img[-1]) == 0:
        img = img[:-1]
  except Exception as e:
    print(f"error_3 {e}")
    return img_or
  try:
    while np.sum(img[:,-1]) == 0:
        img = np.delete(img,-1,1)
  except Exception as e:
    print(f"error_4 {e}")
    return img_or
  
  rows,cols = img.shape
  if rows > cols:
    try:
      factor = 20.0/rows
      rows = 20
      cols = int(round(cols*factor))
      img = cv2.resize(img, (cols,rows))
    except Exception as e:
      print(f"error resizeing..{e}")
      return img_or
  else:
    try:
      factor = 20.0/cols
      cols = 20
      rows = int(round(rows*factor))
      img = cv2.resize(img, (cols, rows))
    except Exception as e:
      print(f"error resizeing..{e}")
      return img_or

  colsPadding = (int(math.ceil((24-cols)/2.0)),int(math.floor((24-cols)/2.0)))
  rowsPadding = (int(math.ceil((24-rows)/2.0)),int(math.floor((24-rows)/2.0)))

  img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')

  img = np.lib.pad(img,(rowsPadding,colsPadding),"constant")

  shiftx,shifty = getBestShift(img)
  shifted = shift(img,shiftx,shifty)

  img_n = create_3dimage(numpy_i, shifted)
  
  print("IMAGE SHAPE IN PREPRO ", img_n.shape)
  """
  img_n = create_3dimage(numpy_i, img)

  return img_n


def create_3dimage(input_np, shiff_np):
  img_or_3c = input_np.copy()

  img_or_3c = cv2.resize(img_or_3c, (shiff_np.shape[1],shiff_np.shape[0]))

  gray = shiff_np.copy()

  img_n = np.zeros_like(img_or_3c)

  img_n[:,:,0] = gray
  img_n[:,:,1] = gray
  img_n[:,:,2] = gray
  
  #print("IMAGE SHAPE IN PREPRO ", img_n.shape)

  return img_n


def imcrop_tosquare(img):
    """Make any image a square image.
    Parameters
    ----------
    img : np.ndarray
        Input image to crop, assumed at least 2d.
    Returns
    -------
    crop : np.ndarray
        Cropped image.
    """
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop
