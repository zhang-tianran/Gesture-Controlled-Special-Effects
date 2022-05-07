import cv2
import mediapipe as mp
import numpy as np
import math
import os
from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12

bg_image = cv2.imread("ostrich.jpg")

# def replace_background(fg, bg):
#     # bg_image = cv2.imread("sloth.jpg")
#     # bg_image = cv2.imread(fg_path)
#     # frame = cv2.imread(bg_path)
#     bg_image = bg
#     frame = fg

#     # initialize mediapipe
#     mp_selfie_segmentation = mp.solutions.selfie_segmentation
#     selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
#             model_selection=1)


#     RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # get the result
#     results = selfie_segmentation.process(RGB)
#     # extract segmented mask
#     mask = results.segmentation_mask
#     # return mask

#     # show outputs
#     # cv2.imshow("mask", mask)
#     # cv2.imshow("Frame", frame)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     # it returns true or false where the condition applies in the mask
#     condition = np.stack(
#             (results.segmentation_mask,) * 3, axis=-1) > 0.5
#     height, width = frame.shape[:2]
#     # resize the background image to the same size of the original frame
#     bg_image = cv2.resize(bg_image, (width, height))
#     # combine frame and background image using the condition
#     output_image = np.where(condition, frame, bg_image)
#     return output_image


def replace_background(fg, bg):
    # bg_image = cv2.imread("sloth.jpg")
    # bg_image = cv2.imread(fg_path)
    # frame = cv2.imread(bg_path)
    bg_image = bg
    frame = fg

    # initialize mediapipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
    #         model_selection=1)
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # get the result
    results = selfie_segmentation.process(RGB)
    # print
    # cv2.imshow("results", results)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # extract segmented mask

    mask = results.segmentation_mask
    mask = cv2.GaussianBlur(mask, (33, 33), 0)
    # return mask

    # it returns true or false where the condition applies in the mask
    condition = np.stack(
        (mask,) * 3, axis=-1) > 0.6
    height, width = frame.shape[:2]
    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))
    # bg_image = cv2.GaussianBlur(bg_image, (55, 55), 0)
    # combine frame and background image using the condition
    output_image = np.where(condition, frame, bg_image)

    # frame = fg
    # image =fg
    # blurred_img = cv2.GaussianBlur(fg, (21, 21), 0)
    # mask = np.zeros(fg.shape, np.uint8)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(mask, contours, -1, (255,255,255),5)
    # height, width = frame.shape[:2]
    # bg_image = cv2.resize(bg, (width, height))
    # output_image = np.where(mask==np.array([255, 255, 255]), blurred_img, bg)

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
    # resize the background image to the same size of the original frame
    # bg_image = np.zeros((rows, columns,3))
    # bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    # global bg_image
    # bg_image = cv2.resize(bg_image, (columns, rows))
    output_image = np.where(condition, frame, 0)
    return mask, output_image

# turn into ostrich
    # images = [fg, bg]
    # mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # height, width = fg.shape[:2]
    # bg_image = cv2.resize(bg, (width, height))
    # with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
    #     for image in images:
    #         # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
    #         results = selfie_segmentation.process(
    #             cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #         blurred_image = cv2.GaussianBlur(image, (55, 55), 0)
    #         condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    #         output_image = np.where(condition, image, blurred_image)
