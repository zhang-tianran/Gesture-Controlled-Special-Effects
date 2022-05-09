from tkinter.messagebox import NO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras_segmentation.pretrained import pspnet_50_ADE_20K

# model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset

model = None

def get_segmented_object(seg, img, point):
    color = np.array(seg[point[0], point[1], :])

    # note that you can define a color range

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


# segment_image_orig(model, cv2.imread("bedroom2.jpg"))
# segment_image(cv2.imread("bedroom2.jpg"))
