import cv2


def norm_image(self, image):
    """
    Norm image to constant shape
    """
    img = cv2.resize(image, (2500, 1600))
    return img
