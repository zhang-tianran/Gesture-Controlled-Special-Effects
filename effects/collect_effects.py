import cv2 as cv
import numpy as np

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