#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
from collections import deque

from skimage import img_as_float32

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

from utils.helpers import *
from effects.selfie_segmentation import segment_selfie
from effects.gen_segmentation import segment_image, get_segmented_object
from effects.point_art import *
from effects.collect_effects import *

import tensorflow as tf
import tensorflow_hub as hub

G_seg_image = None
seg_object = None
pickup_point = None
placement_point = None
G_mask = None
seg_mode = False
selfie_seg_mode = True

selection_modes = {
    "select": 0,
    "drawing": 1,
    "effect": 2,
    "segmentation": 3,
    "panoroma": 4,
    "tunnel": 5,
}


def display_selection_mode(selection_mode, display_text):
    for a_key in selection_modes:
        if (selection_mode == selection_modes["select"]):
            display_text += "1. drawing\n2. graphic effects\n3. segmentation\n4. panaroma\n5. light tunnel\n"
            break
        elif (selection_mode == selection_modes["effect"]):
            text = "1. ghibli\n2. cartoon\n3. point art\n4. avatar\n"
            display_text = text + display_text
            break
        elif selection_mode == selection_modes[a_key]:
            text = a_key + "\n"
            display_text = text + display_text
            break
    return display_text


def add_text(frame, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 3

    y0, dy = 240, 80
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv.putText(frame, line, (50, y), font, fontScale, color, thickness)

    return frame


def stylization_popup(stylization_model, frame, style_image):
    temp_debug_image = frame
    temp_debug_image = tf.expand_dims(temp_debug_image, 0)
    temp_debug_image = img_as_float32(temp_debug_image)
    temp_debug_image = tf.convert_to_tensor(temp_debug_image)

    img = stylization_model(temp_debug_image, style_image)
    img = np.asarray(img[0][0])
    cv.imshow("stylization", img)


def impressionism_popup(frame):
    impressionism = run_impressionistic_filter(frame, False)
    cv.imshow("impressionism", impressionism)


def place_segmentation(debug_image):
    if seg_object is not None and pickup_point is not None and placement_point is not None:
        difference = np.array(placement_point) - np.array(pickup_point)
        shift_y = int(difference[1])  # col
        shift_x = int(difference[0])  # row
        if shift_x > 0:
            start_col = 0
            end_col = debug_image.shape[1] - shift_x
            start_col_debug = shift_x
            end_col_debug = debug_image.shape[1]
        else:
            start_col = abs(shift_x)
            end_col = debug_image.shape[1]
            start_col_debug = 0
            end_col_debug = debug_image.shape[1] - abs(shift_x)
        if shift_y < 0:
            start_row = abs(shift_y)
            end_row = debug_image.shape[0]
            start_row_debug = 0
            end_row_debug = debug_image.shape[0] - abs(shift_y)
        else:
            start_row = 0
            end_row = debug_image.shape[0] - abs(shift_y)
            start_row_debug = abs(shift_y)
            end_row_debug = debug_image.shape[0]

        base_seg = np.zeros(
            (debug_image.shape[0], debug_image.shape[1], 3))
        rel_seg_obj = seg_object[start_row:end_row,
                                 start_col:end_col, :]
        base_seg[start_row_debug:end_row_debug,
                 start_col_debug:end_col_debug, :] = rel_seg_obj
        G_mask_temp = G_mask[start_row:end_row, start_col:end_col]

        condition = np.stack((G_mask_temp,) * 3, axis=-1) > 0.6

        debug_image[start_row_debug:end_row_debug,
                    start_col_debug:end_col_debug, :] = np.where(condition, rel_seg_obj, debug_image[start_row_debug:end_row_debug,
                                                                                                     start_col_debug:end_col_debug, :])
        return debug_image


def main():

    global G_seg_image
    global seg_object
    global placement_point
    global pickup_point
    global G_mask
    global selfie_seg_mode
    global seg_mode

    # camera preparation ###############################################################
    cap = cv.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    keypoint_classifier = KeyPointClassifier()

    # read models ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # modes setup ###########################################################
    stylization_model = hub.load("model/image_stylization")
    style_image_og = cv.cvtColor(
        cv.imread("assets/ghibli-style.png"), cv.COLOR_BGR2RGB)
    style_image_og = img_as_float32(style_image_og)
    style_image_og = tf.expand_dims(style_image_og, 0)

    panorama = cv.imread('assets/panorama.png')
    view_start = 0
    view_shift_speed = 1000
    view_width = 5000
    panorama_height, panorama_width, _ = panorama.shape

    canvas = np.zeros((1, 1, 3))
    in_mode = False

    # point & gesture history generation #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    selection_mode = selection_modes["select"]
    frame_num = 0

    while True:
        display_text = ""
        frame_num += 1

        # exit the program #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # capture image #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # check output #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # recoginization ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # process landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # process hand sign id
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if (hand_sign_id == 0):
                    hand_sign_id = -1
                elif (hand_sign_id == 6):
                    hand_sign_id = 0
                elif (hand_sign_id == 1):
                    point_history.append(landmark_list[8])

                # mode selection ####################################################################
                if ((selection_mode == selection_modes["select"] and hand_sign_id != 0)):

                    if (hand_sign_id < len(selection_modes)):
                        in_mode = False
                        selection_mode = hand_sign_id
                    else:
                        hand_sign_id = 0
                        selection_mode = selection_mode["select"]
                elif (hand_sign_id == 0):
                    # clear
                    if (frame_num % 50 < 12):
                        #  display_text += "Entered selection mode!\nChoose a mode\n"
                        selection_mode = selection_modes["select"]
                        G_seg_image = None
                        seg_object = None
                        pickup_point = None
                        placement_point = None
                        G_mask = None
                        seg_mode = False
                        selfie_seg_mode = True
                        in_mode = False
                        try:
                            cv.destroyWindow("panorama-view")
                            cv.destroyWindow("stylization")
                            cv.destroyWindow("impressionism")
                        except Exception as e:
                            raise e
                # Entering modes
                else:
                    if selection_mode == selection_modes["tunnel"]:
                        debug_image = tunnel_effect(
                            debug_image, landmark_list[9])
                    elif selection_mode == selection_modes["effect"]:
                        if (hand_sign_id == 1):  # ghibli stylization
                            stylization_popup(
                                stylization_model, debug_image, style_image_og)
                        elif (hand_sign_id == 2):  # cartoon
                            debug_image = cartoon_effect(debug_image, False)
                        elif (hand_sign_id == 3):  # point art stylization
                            impressionism_popup(debug_image)
                        elif (hand_sign_id == 4):  # avatar blue skin
                            debug_image = cartoon_effect(
                                debug_image, color_change=True)
                    elif selection_mode == selection_modes["panoroma"]:
                        if hand_sign_id == 1:
                            if (landmark_list[8][0] > point_history[-2][0]):
                                view_start += view_shift_speed
                            else:
                                view_start -= view_shift_speed
                            view_start = min(
                                max(0, view_start), panorama_width - view_width)
                        panorama_in_view = panorama[:,
                                                    view_start:view_start+view_width]
                        cv.imshow('panorama-view', panorama_in_view)
                    elif selection_mode == selection_modes["segmentation"]:
                        if hand_sign_id == 3:
                            display_text += "2: Selfie Segmentation\n4: General Segmentation \n"
                            display_text += "(Note: pause for a bit if choose 4)\n"
                        else:
                            display_text += "Slide segmented sticker around with 1\n"
                            display_text += "Set sticker position temporarily by holding 5\n" 
                            display_text += "Reset with 6! \n"                    
                        if hand_sign_id == 1 and G_seg_image is not None and seg_object is not None:
                            placement_point = [min(max(0, landmark_list[8][0]), debug_image.shape[1]), min(
                                max(0, landmark_list[8][1]), debug_image.shape[0])]
                            debug_image = place_segmentation(debug_image)
                        elif hand_sign_id == 2 and G_seg_image is None:
                            G_mask, G_seg_image = segment_selfie(
                                debug_image)
                            pickup_point = [min(max(0, landmark_list[8][0]), debug_image.shape[1]), min(
                                max(0, landmark_list[8][1]), debug_image.shape[0])]
                            seg_object = G_seg_image
                        elif hand_sign_id == 4 and G_seg_image is None:
                            G_seg_image = segment_image(debug_image)
                            pickup_point = [min(max(0, landmark_list[8][0]), debug_image.shape[1]), min(
                                max(0, landmark_list[8][1]), debug_image.shape[0])]
                            G_mask, seg_object = get_segmented_object(
                                G_seg_image, debug_image, pickup_point)
                        elif hand_sign_id == 5 and seg_object is not None and pickup_point is not None and placement_point is not None:
                            debug_image = place_segmentation(debug_image)
                    elif selection_mode == selection_modes["drawing"]:
                        display_text += "Clear the canvas with 5!\n"
                        h, w, c = debug_image.shape
                        if hand_sign_id == 5:
                            canvas = np.zeros((h, w, c))
                        else:
                            in_mode = True
                            canvas = cv.resize(canvas, (w, h))
                            canvas = drawing(canvas, point_history)
        
                # generate information ####################################################################
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    ""
                )
        else:
            point_history.append([0, 0])

        # add text
        display_text = display_selection_mode(selection_mode, display_text)
        add_text(debug_image, display_text)

        # show image
        if (in_mode):
            final = cv.addWeighted(canvas.astype(
                'uint8'), 1, debug_image, 1, 0)
            cv.imshow('Hand Gesture Recognition', final)
        else:
            cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
