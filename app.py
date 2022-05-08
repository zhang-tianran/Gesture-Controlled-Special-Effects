#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
from collections import Counter
from collections import deque

from skimage import img_as_float32


import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier
from model import PointHistoryClassifier

from KazuhitoTakahashiUtils.helpers import *
from selfie_segmentation import replace_background, segment_selfie
from gen_segmentation import segment_image, get_segmented_object
# from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12
# from keras_segmentation.predict import predict

import tensorflow as tf
import tensorflow_hub as hub

from point_art import *

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
    selection_mode_found = False
    for a_key in selection_modes:
        if (selection_mode == selection_modes["effect"]):
            display_text += "1. ghibli\n2. cartoon\n3. point art\n4. avatar\n"
            break

        elif selection_mode == selection_modes[a_key]:
            display_text += (a_key + "\n")
            selection_mode_found = True
            break
    if not selection_mode_found:
        display_text += "Selection mode not found\n"

    return display_text


def add_text(frame, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    pos = (100, 200)
    org = (50, 50)
    fontScale = 2
    color = (255, 0, 0)
    thickness = 2

    y0, dy = 240, 80
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv.putText(frame, line, (50, y), font, fontScale, color, thickness)

    #  cv.putText(frame, text, pos, font,
    #                 fontScale, color, thickness, cv.LINE_AA)
    return frame


def cartoon_effect(frame, color_change):
    # prepare color

    if color_change:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    img_color = cv.pyrDown(cv.pyrDown(frame))
    for _ in range(3):
        img_color = cv.bilateralFilter(img_color, 9, 9, 7)
    img_color = cv.pyrUp(cv.pyrUp(img_color))

    # prepare edges
    img_edges = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    img_edges = cv.adaptiveThreshold(
        cv.medianBlur(img_edges, 7), 255,
        cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
        9, 2,)
    img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2RGB)

    # combine color and edges
    frame = cv.bitwise_and(img_color, img_edges)
    return frame


def tunnel_effect(image, landmark):
    (h, w) = image.shape[:2]
    center = np.array([landmark[0], landmark[1]])
    radius = h / 2.5

    i, j = np.mgrid[0:h, 0:w]
    xymap = np.dstack([j, i]).astype(np.float32)  # "identity" map

    # coordinates relative to center
    coords = (xymap - center)
    # distance to center
    dist = np.linalg.norm(coords, axis=2)
    # touch only what's outside of the circle
    mask = (dist >= radius)
    # project onto circle (calculate unit vectors, move onto circle, then back to top-left origin)
    xymap[mask] = coords[mask] / dist[mask, None] * radius + center

    out = cv.remap(image, map1=xymap, map2=None, interpolation=cv.INTER_LINEAR)
    return out


def drawing(image, point_history):
    pre = None
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            if pre == None:
                pre = point
            else:
                cv.line(image, pre, point, (200, 140, 30), 2)
                pre = point
    return image


def stylization_popup(stylization_model, frame, style_image):
    temp_debug_image = frame
    temp_debug_image = tf.expand_dims(temp_debug_image, 0)
    temp_debug_image = img_as_float32(temp_debug_image)
    temp_debug_image = tf.convert_to_tensor(temp_debug_image)

    hello = stylization_model(temp_debug_image, style_image)
    hello = np.asarray(hello[0][0])
    cv.imshow("stylization", hello)


def impressionism_popup(frame):
    impressionism = run_impressionistic_filter(frame, False)
    cv.imshow("impressionism", impressionism)


def place_segmentation(debug_image):
    if seg_object is not None and pickup_point is not None and placement_point is not None:
        difference = np.array(placement_point) - np.array(pickup_point)
        print(difference)
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
        print("YO")
        print(rel_seg_obj.shape)
        print(G_mask.shape)
        print(rel_seg_obj.shape)
        print(debug_image.shape)
        G_mask_temp = G_mask[start_row:end_row,
                             start_col:end_col]
        print("HIIII")
        print(G_mask.shape)
        # THIS WAS THE ORIGINAL
        condition = np.stack((G_mask_temp,) * 3, axis=-1) > 0.6
        # height, width = output.shape[:2]
        # resize the background image to the same size of the original frame
        # bg_image = cv2.resize(bg_image, (width, height))
        # # this was iffy:
        debug_image[start_row_debug:end_row_debug,
                    start_col_debug:end_col_debug, :] = np.where(condition, rel_seg_obj, debug_image[start_row_debug:end_row_debug,
                                                                                                     start_col_debug:end_col_debug, :])
        return debug_image


def main():

    in_mode = False

    global G_seg_image
    global seg_object
    global placement_point
    global pickup_point
    global G_mask
    global selfie_seg_mode
    global seg_mode
    # global canvas

    use_brect = True

    # camera preparation ###############################################################
    cap = cv.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    #  if (panorama_mode):
    panorama = cv.imread('assets/panorama.png')
    view_start = 0
    view_shift_speed = 1000
    view_width = 5000
    panorama_height, panorama_width, _ = panorama.shape
    #  view_shift_speed = 400

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    canvas = np.zeros((1, 1, 3))

    # read models ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    stylization_model = hub.load("model/image_stylization")
    style_image_og = cv.cvtColor(
        cv.imread("assets/ghibli-style.png"), cv.COLOR_BGR2RGB)
    style_image_og = img_as_float32(style_image_og)
    style_image_og = tf.expand_dims(style_image_og, 0)

    # point & gesture history generation #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

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

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # ランドマークの計算
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 相対座標・正規化座標への変換
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if (hand_sign_id == 0):
                    hand_sign_id = -1

                if (hand_sign_id == 6):
                    hand_sign_id = 0
                if hand_sign_id == 1:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                #  print("hand_sign_id: ", hand_sign_id)

                #  if (hand_sign_id == 0):
                #      view_start += view_shift_speed
                #      pyautogui.scroll(-5)
                #  elif (hand_sign_id == 1):
                #      view_start -= view_shift_speed
                #      pyautogui.scroll(5)

                #  print(frame_num)
                if (selection_mode == selection_modes["select"] and hand_sign_id != 0):
                    in_mode = False
                    selection_mode = hand_sign_id
                elif (hand_sign_id == 0):
                    if (frame_num % 50 < 12):
                        # clear
                        display_text += "Entered selection mode!\nChoose a mode\n"
                        selection_mode = selection_modes["select"]
                        G_seg_image = None
                        seg_object = None
                        pickup_point = None
                        placement_point = None
                        G_mask = None
                        seg_mode = False
                        selfie_seg_mode = True
                        try:
                            cv.destroyWindow("panorama-view")
                            cv.destroyWindow("stylization")
                            cv.destroyWindow("impressionism")
                        except Exception as e:
                            raise e
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
                        if hand_sign_id == 2:
                            if (landmark_list[8][0] > point_history[-3][0]):
                                print("right")
                                view_start += view_shift_speed
                            else:
                                print("left")
                                view_start -= view_shift_speed
                            view_start = min(
                                max(0, view_start), panorama_width - view_width)
                        panorama_in_view = panorama[:,
                                                    view_start:view_start+view_width]
                        cv.imshow('panorama-view', panorama_in_view)
                    elif selection_mode == selection_modes["segmentation"]:
                        # if hand_sign_id == 3 and selfie_seg_mode == True:
                        #     if G_seg_image is None:
                        #         G_mask, G_seg_image = segment_selfie(
                        #             debug_image)
                        # if hand_sign_id == 1 and G_seg_image is not None and seg_object is None and selfie_seg_mode == True:
                        #     print(landmark_list[8])
                        #     print("HIHIHHI")
                        #     pickup_point = landmark_list[8]
                        #     seg_object = G_seg_image
                        if hand_sign_id == 2:
                            if G_seg_image is None:
                                G_mask, G_seg_image = segment_selfie(
                                    debug_image)
                                pickup_point = landmark_list[8]
                                seg_object = G_seg_image
                        if hand_sign_id == 4:
                            if G_seg_image is None:
                                G_seg_image = segment_image(debug_image)
                                pickup_point = landmark_list[8]
                                G_mask, seg_object = get_segmented_object(
                                    G_seg_image, debug_image, pickup_point)
                    elif selection_mode == selection_modes["drawing"]:
                        h, w, c = debug_image.shape
                        if hand_sign_id == 5: 
                            canvas = np.zeros((h, w, c))
                        else: 
                            in_mode = True
                            canvas = cv.resize(canvas, (w, h))
                            canvas = drawing(canvas, point_history)


                        # if hand_sign_id == 3:
                        #     selfie_seg_mode = True
                        #     seg_mode = False
                        #     if hand_sign_id == 3 and selfie_seg_mode == True:
                        #         if G_seg_image is None:
                        #             G_mask, G_seg_image = segment_selfie(
                        #                 debug_image)
                        #     if hand_sign_id == 1 and G_seg_image is not None and seg_object is None and selfie_seg_mode == True:
                        #         print(landmark_list[8])
                        #         print("HIHIHHI")
                        #         pickup_point = landmark_list[8]
                        #         seg_object = G_seg_image
                        # if hand_sign_id == 4:
                        #     selfie_seg_mode = False
                        #     seg_mode = True
                        #     if hand_sign_id == 3 and seg_mode == True:
                        #         if G_seg_image is None:
                        #             G_seg_image = segment_image(debug_image)
                        #     if hand_sign_id == 1 and G_seg_image is not None and seg_object is None and seg_mode == True:
                        #         print(landmark_list[8])
                        #         pickup_point = landmark_list[8]
                        #         G_mask, seg_object = get_segmented_object(
                        #             G_seg_image, debug_image, pickup_point)
                        if hand_sign_id == 1 and G_seg_image is not None and seg_object is not None:
                            placement_point = landmark_list[8]
                            print("PLACE")
                            print(landmark_list[8])
                            debug_image = place_segmentation(debug_image)
                        if hand_sign_id == 5 and seg_object is not None and pickup_point is not None and placement_point is not None:
                            print("AHHHHHH")
                            debug_image = place_segmentation(debug_image)
        

                # gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # 直近検出の中で最多のジェスチャーIDを算出
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # generate information
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        display_text = display_selection_mode(selection_mode, display_text)
        add_text(debug_image, display_text)

        # show image #############################################################
        if (in_mode): 
            final = cv.addWeighted(canvas.astype('uint8'), 1, debug_image, 1, 0)
            cv.imshow('Hand Gesture Recognition', final)
        else: 
            cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
