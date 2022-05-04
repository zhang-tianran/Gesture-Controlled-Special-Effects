"""
Final Project by Team Strange
CS1430 - Computer Vision
Brown University

Reference Paper:
https://web.stanford.edu/class/ee368/Project_Autumn_1516/Reports/Hong_Liu.pdf

"""
import numpy as np
import cv2
import random
from scipy.spatial import distance
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

RADIUS = 6
NUM_COLORS = 10
THICKNESS = -1
MAX_X = 200
MAX_Y = 200
STRIDE = 4 # better than 2 or 3

def apply_low_pass(img):
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)
    return img


def downsample_image(img):
    scale_percent = 0.6
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dims = (width, height)
    return cv2.resize(img, dims, interpolation=cv2.INTER_AREA)


def find_primary_palette(downsampled_img):
    # use KMeans
    clt = KMeans(n_clusters=NUM_COLORS)
    clt.fit(downsampled_img.reshape(-1, 3))
    ret = clt.cluster_centers_
    # should be of shape (NUM_COLORS, 3)
    return ret 


def add_complements(palette):
    complements = 255 - palette
    palette = np.vstack((palette, complements))
    return palette


def create_blank_canvas(img_x, img_y):
    canvas = np.zeros((img_x, img_y, 3), np.uint8)
    canvas[:, :] = (255, 255, 255)
    return canvas


def add_slight_shifts(w, h, blurry):
    img_coords = []
    for row_val in range(0, h, STRIDE):
        for col_val in range(0, w, STRIDE):
            # experimented with shift values
            x_slight_shift = random.randint(-1, 2)
            y_slight_shift = random.randint(-1, 1)
            col = x_slight_shift + col_val
            row = y_slight_shift + row_val
            if (col < w and row < h):
                img_coords.append((row, col))
            else:
                img_coords.append((row % h, col % w))

    if not blurry:
        random.shuffle(img_coords)
    return img_coords


def get_colors_representing_pixels(img, img_coords):
    colors = []
    for coord in img_coords:
        colors.append(img[coord[0], coord[1]])
    return colors


def compute_color_probabilities(pixels, palette):
    # use distance.cdist
    # reference: open source project from https://www.programcreek.com/python/?CodeExample=compute+color
    distances = distance.cdist(pixels, palette)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    
    return distances


def get_colors_in_cluster(cluster_probs, palette):
    probs = np.argsort(cluster_probs)
    color_idx = probs[len(probs) - 1]
    return palette[color_idx]


def paint_dot(canvas, x, y, color):
    cv2.circle(canvas, (x, y), RADIUS, color, THICKNESS) 
    # to improve this, could decrease opacity so paint blends


def run_impressionistic_filter(img, blurry):

    # apply low pass
    img = apply_low_pass(img)

    # find primary palette
    palette = find_primary_palette(downsample_image(img)) # palette: (10, 3)
    
    # palette = convert_rgb_to_hsv(palette) # commented out because colors produced aren't right

    # add complementary colors to palette to increase contrast
    palette = add_complements(palette)

    # create blank canvas
    canvas = create_blank_canvas(img.shape[0], img.shape[1])
    
    img_coords = add_slight_shifts(img.shape[1], img.shape[0], blurry=blurry)
    colors_representing_pixels = get_colors_representing_pixels(img, img_coords)

    # Two colors are chosen in respect to minimum distance to the original
    # pixel’s color in the RGB color space. Then the third color is a
    # randomly chosen color from the remaining 14 colors.

    # get probabilities from clusters
    color_probabilities = compute_color_probabilities(colors_representing_pixels, palette)

    # for each pixel in new image grid, paint a dot
    for i, (y, x) in enumerate(img_coords):
        color = get_colors_in_cluster(color_probabilities[i], palette)
        paint_dot(canvas, x, y, color)
       
    return canvas
    
# For video style transfer, call:
# img = run_impressionistic_filter(img, False)

# For image style transfer, call:
filepath = "pavillion.jpeg"
outpath = "pavillion-output.jpeg"
car_window_view_outpath = "pavillion-output-car-window-view.jpeg"

img = cv2.imread(filepath)

# effect: impressionism
img = run_impressionistic_filter(img, False)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite(outpath, img)

# effect: rainy day car window view
img = run_impressionistic_filter(img, True)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite(car_window_view_outpath, img)