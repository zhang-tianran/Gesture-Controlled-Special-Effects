import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12

model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset

# load the pretrained model trained on Cityscapes dataset


model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset

# THIS WORKS FOR TESTING


# def segment_image_orig(model, img):
#     out = model.predict_segmentation(
#         inp=img
#     )
#     dim = (img.shape[1], img.shape[0])
#     # print(dim)
#     ret_val = out.astype('uint8')
#     ret_val = cv2.resize(ret_val, dim, interpolation=cv2.INTER_AREA)
#     ret_val_color = ret_val
#     print(ret_val.shape)
#     print(img.shape)
#     hsv = cv2.applyColorMap(ret_val, cv2.COLORMAP_HSV)
#     # Convert BGR to HSV
#     # hsv = cv2.cvtColor(ret_val_color, cv2.COLOR_BGR2HSV)
#     color = np.array(hsv[200, 200, :])

#     # # define blue color range
#     # light_blue = np.array([110, 50, 50])
#     # dark_blue = np.array([130, 255, 255])

#     # Threshold the HSV image to get only blue colors
#     mask = cv2.inRange(hsv, color, color)

#     # Bitwise-AND mask and original image
#     output = cv2.bitwise_and(img, img, mask=mask)

#     cv2.imshow("Color Detected", np.hstack((hsv, output)))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # filt_image = np.where(ret_val_color == color, img, 0)

#     cv2.imshow("out", hsv)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # cv2.imshow("out", filt_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     #=-----------------------------------------
#     # RGB = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
#     # # get the result
#     # results = selfie_segmentation.process(RGB)

#     # mask = results.segmentation_mask
#     # mask = cv2.GaussianBlur(mask, (33, 33), 0)

#     # it returns true or false where the condition applies in the mask
#     bg_image = cv2.imread("ostrich.jpg")
#     condition = np.stack(
#         (mask,) * 3, axis=-1) > 0.1
#     height, width = output.shape[:2]
#     # resize the background image to the same size of the original frame
#     bg_image = cv2.resize(bg_image, (width, height))
#     # bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
#     # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
#     # bg_image = cv2.GaussianBlur(bg_image, (55, 55), 0)
#     # combine frame and background image using the condition
#     output_image = np.where(condition, output, bg_image)
#     # height, width = frame.shape[:2]
#     # resize the background image to the same size of the original frame
#     # bg_image = cv2.resize(bg_image, (width, height))
#     # bg_image = cv2.GaussianBlur(bg_image, (55, 55), 0)
#     # combine frame and background image using the condition

#     cv2.imshow("out", output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return ret_val

# THIS WORKS FOR TESTING


def get_segmented_object(seg, img, point):
    # out = model.predict_segmentation(
    #     inp=img
    # )
    # dim = (img.shape[1], img.shape[0])
    # # print(dim)
    # ret_val = out.astype('uint8')
    # ret_val = cv2.resize(ret_val, dim, interpolation=cv2.INTER_AREA)
    # ret_val_color = ret_val
    # print(ret_val.shape)
    # print(img.shape)
    # hsv = cv2.applyColorMap(ret_val, cv2.COLORMAP_HSV)
    # # Convert BGR to HSV
    # # hsv = cv2.cvtColor(ret_val_color, cv2.COLOR_BGR2HSV)
    color = np.array(seg[point[0], point[1], :])

    # # define blue color range
    # light_blue = np.array([110, 50, 50])
    # dark_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(seg, color, color)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("Color Detected", np.hstack((hsv, output)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # filt_image = np.where(ret_val_color == color, img, 0)

    # cv2.imshow("out", hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # cv2.imshow("out", filt_image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
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
    # print(ret_val.shape)
    # print(img.shape)
    hsv = cv2.applyColorMap(ret_val, cv2.COLORMAP_HSV)
    # # Convert BGR to HSV
    # # hsv = cv2.cvtColor(ret_val_color, cv2.COLOR_BGR2HSV)
    # color = np.array(hsv[200, 200, :])

    # # # define blue color range
    # # light_blue = np.array([110, 50, 50])
    # # dark_blue = np.array([130, 255, 255])

    # # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, color, color)

    # # Bitwise-AND mask and original image
    # output = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("Color Detected", np.hstack((hsv, output)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # filt_image = np.where(ret_val_color == color, img, 0)

    # cv2.imshow("out", hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # cv2.imshow("out", filt_image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    return hsv


# def segment_image_orig(model, img):
#     # model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

#     # model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

#     # model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

#     # load any of the 3 pretrained models

#     # # out = model.predict_segmentation(
#     # #     inp=cv2.imread("bedroom2.jpg"),
#     # #     out_fname="out2.png"
#     # # )
#     # out = model.predict_segmentation(
#     #     inp=img,
#     #     out_fname="out2.png"
#     # )
#     out = model.predict_segmentation(
#         inp=img
#     )
#     ret_val = np.zeros((473, 473,3))
#     out = out/255
#     ret_val[:,:,0] = out * 2
#     # ret_val[:,:,1] = out
#     dim = (img.shape[1], img.shape[0])
# #     # print(dim)
# #     ret_val = out.astype('uint8')
#     ret_val = cv2.resize(ret_val, dim, interpolation=cv2.INTER_AREA)
#     ret_val_color = ret_val
#     color = np.array(ret_val[1500,1500])
#     print(tuple(color))


#     # # Convert BGR to HSV
#     # hsv = cv2.cvtColor(ret_val_color, cv2.COLOR_BGR2HSV)

#     # # # define blue color range
#     # # light_blue = np.array([110, 50, 50])
#     # # dark_blue = np.array([130, 255, 255])

#     # # Threshold the HSV image to get only blue colors
#     # mask = cv2.inRange(hsv, color, color)

#     # # Bitwise-AND mask and original image
#     # output = cv2.bitwise_and(ret_val_color, ret_val_color, mask=mask)

#     # cv2.imshow("Color Detected", np.hstack((ret_val_color, output)))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     filt_image = np.where(ret_val_color == color, img, 0)

#     # ret_val[:,:,2] = out
#     # ret_val = np.asarray(out)
#     # ret_val = np.stack((r,b,r), )
#     # ret_val = PIL.Image.open(out)
#     # print(out)
#     # print(np.shape(r))
#     # print(np.shape(b))
#     # print(np.shape(out))
#     # print(np.shape(ret_val))
#     # print(type(model))
#     # plt.imshow(ret_val)
#     # plt.show()

#     # # # # print(out)
#     # # out = out.astype('uint8')
#     # # print(np.shape(out))
#     # # print(np.shape(out[0]))
#     # cv2.imshow("out", ret_val)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()]
#     cv2.imshow("out", ret_val_color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imshow("out", filt_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return ret_val

# # interesting affect but not exactly what i want
# def segment_image(model, img):
#     out = model.predict_segmentation(
#         inp=img
#     )
#     r = out[0, :]
#     b = out[1, :]
#     # ret_val = np.zeros((473, 473, 3))
#     # out = out/255
#     # ret_val[:, :, 0] = out * 2
#     # ret_val[:, :, 1] = out*2

#    # with no skew/stretch resize
#     # width = int (out.shape[1] * (int(img.shape[1]/out.shape[1]) * 100) / 100)
#     # height = int(out.shape[0] * (int(img.shape[0]/out.shape[0]) * 100) / 100)
#     # dim = (width, height)
#     dim = (img.shape[1], img.shape[0])
#     # print(dim)
#     ret_val = out.astype('uint8')
#     ret_val = cv2.resize(ret_val, dim, interpolation=cv2.INTER_AREA)
#     # print(ret_val.shape)
#     # print(img.shape)
#     # ret_val = ret_val.astype('uint8')
#     # applyColorMap(img_in, img_color, COLORMAP_JET)
#     # im_gray = cv2.imread("pluto.jpg", cv2.IMREAD_GRAYSCALE)
#     ret_val_color = cv2.applyColorMap(ret_val, cv2.COLORMAP_HSV)
#     colorG = ret_val_color[1500, 1500, 0]
#     colorB = ret_val_color[1500, 1500, 1]
#     colorR = ret_val_color[1500, 1500, 2]
#     color = (colorG, colorB, colorR)
#     print(color)
#     filt_image = np.where(ret_val_color == tuple(color), img, 0)
#     cv2.imshow("out", ret_val_color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imshow("out", filt_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return ret_val

# # def segment_image(model, img):
# #     out = model.predict_segmentation(
# #         inp=img
# #     )
# #     r = out[0, :]
# #     b = out[1, :]
# #     # ret_val = np.zeros((473, 473, 3))
# #     # out = out/255
# #     # ret_val[:, :, 0] = out * 2
# #     # ret_val[:, :, 1] = out*2

# #    # with no skew/stretch resize
# #     # width = int (out.shape[1] * (int(img.shape[1]/out.shape[1]) * 100) / 100)
# #     # height = int(out.shape[0] * (int(img.shape[0]/out.shape[0]) * 100) / 100)
# #     # dim = (width, height)
# #     dim = (img.shape[1], img.shape[0])
# #     # print(dim)
# #     ret_val = out.astype('uint8')
# #     ret_val = cv2.resize(ret_val, dim, interpolation=cv2.INTER_AREA)
# #     # # print(ret_val.shape)
# #     # # print(img.shape)
# #     # # ret_val = ret_val.astype('uint8')
# #     # # applyColorMap(img_in, img_color, COLORMAP_JET)
# #     # # im_gray = cv2.imread("pluto.jpg", cv2.IMREAD_GRAYSCALE)
# #     # ret_val_color = cv2.applyColorMap(ret_val, cv2.COLORMAP_HSV)
# #     # colorG = ret_val_color[1500, 1500, 0]
# #     # colorB = ret_val_color[1500, 1500, 1]
# #     # colorR = ret_val_color[1500, 1500, 2]
# #     # color = (colorG, colorB, colorR)
# #     ret_val_color = ret_val
# #     color = ret_val_color[1500, 1500]
# #     print(color)
# #     filt_image = np.where(ret_val_color == color, ret_val_color, 0)
# #     cv2.imshow("out", ret_val_color)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #     cv2.imshow("out", filt_image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# #     return ret_val


# segment_image_orig(model, cv2.imread("bedroom2.jpg"))
# segment_image(cv2.imread("bedroom2.jpg"))
