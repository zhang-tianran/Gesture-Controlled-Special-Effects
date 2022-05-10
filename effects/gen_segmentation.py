import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras_segmentation.pretrained import model_from_checkpoint_path
import tensorflow.keras as keras


def pspnet_50_ADE_20K(): 
    model_config = {
            "input_height": 473,
            "input_width": 473,
            "n_classes": 150,
            "model_class": "pspnet_50",
            }

    model_url = "https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1"

    latest_weights  = "/Users/ztr/.keras/models/pspnet50_ade20k.h5"
    
    return model_from_checkpoint_path(model_config, latest_weights)

model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset

def get_segmented_object(seg, img, point):
    color = np.array(seg[point[1], point[0], :])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(seg, color, color)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(img, img, mask=mask)
    return mask, output

def segment_image(img):
    out = model.predict_segmentation(
        inp=img
    )
    dim = (img.shape[1], img.shape[0])
    ret_val = out.astype('uint8')
    ret_val = cv2.resize(ret_val, dim, interpolation=cv2.INTER_AREA)
    hsv = cv2.applyColorMap(ret_val, cv2.COLORMAP_HSV)
    return hsv