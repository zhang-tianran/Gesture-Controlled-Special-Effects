"""
Final Project by Team Strange
CS1430 - Computer Vision
Brown University

Impressionism Style Transfer

Reference Paper:
https://web.stanford.edu/class/ee368/Project_Autumn_1516/Reports/Hong_Liu.pdf

"""
import numpy as np
import cv2
from scipy.spatial import distance
from sklearn.cluster import KMeans

def apply_low_pass(img):
    pass

def downsample_image(img):
    pass

def apply_blur(img):
    pass

def find_primary_palette(downsampled_img, primary_colors):
    # use KMeans
    pass

def convert_rgb_to_hsv(palette):
    pass

def brighten_and_saturate(palette):
    pass

def add_complements(palette):
    pass

def create_blank_canvas():
    pass

def randomize():
    pass

def pixel_to_dot_cluster(pixels, palette):
    # use distance.cdist
    pass

def get_colors_in_cluster(cluster_probs, palette):
    pass

def paint_dot():
    pass

def main():
    pass
