import cv2
import numpy as np

IMG_SIZE = (224, 224)

def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img