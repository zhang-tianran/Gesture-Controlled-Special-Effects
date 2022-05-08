import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12

model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset




model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset



def get_segmented_object(seg, img, point):
    color = np.array(seg[point[0], point[1], :])

    # # define blue color range
    # light_blue = np.array([110, 50, 50])
    # dark_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(seg, color, color)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(img, img, mask=mask)
    return mask, output


# THIS WORKS FOR TESTING
def segment_image(img):
    out = model.predict_segmentation(
        inp=img
    )
    dim = (img.shape[1], img.shape[0])
    # print(dim)
    ret_val = out.astype('uint8')
    ret_val = cv2.resize(ret_val, dim, interpolation=cv2.INTER_AREA)
    ret_val_color = ret_val
    hsv = cv2.applyColorMap(ret_val, cv2.COLORMAP_HSV)
    return hsv


# segment_image_orig(model, cv2.imread("bedroom2.jpg"))
# segment_image(cv2.imread("bedroom2.jpg"))
