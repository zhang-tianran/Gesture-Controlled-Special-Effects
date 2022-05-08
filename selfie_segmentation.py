import cv2
import mediapipe as mp
import numpy as np
import math
import os
from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12


def replace_background(fg, bg):
    bg_image = bg
    frame = fg

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(RGB)

    mask = results.segmentation_mask
    mask = cv2.GaussianBlur(mask, (33, 33), 0)
    # return mask

    # it returns true or false where the condition applies in the mask
    condition = np.stack(
        (mask,) * 3, axis=-1) > 0.6
    height, width = frame.shape[:2]
    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))
    output_image = np.where(condition, frame, bg_image)

    return output_image


def segment_selfie(fg):
    frame = fg

    # initialize mediapipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(RGB)
    mask = results.segmentation_mask
    mask = cv2.GaussianBlur(mask, (33, 33), 0)
    # return mask

    # it returns true or false where the condition applies in the mask
    condition = np.stack(
        (mask,) * 3, axis=-1) > 0.6
    rows, columns = frame.shape[:2]
    output_image = np.where(condition, frame, 0)
    return mask, output_image

