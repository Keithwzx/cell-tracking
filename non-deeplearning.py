from PIL import Image
import os, os.path
from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from tracking import *



def detector(image):
    """
    This function uses adaptive filtering for segmentation of each cell which is a straigthforward task for such an
    easy background.
    :param image: .png file.
    :return:
    """
    def color_filtering(image):
        """

        :param image:
        :return:
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        th3 = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 51, 6)
        kernel = np.ones((8, 8), np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(closing,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected_contours = []
        for contour in contours:
            if len(contour) > 50:
                selected_contours.append(contour)
        cv2.drawContours(image, selected_contours, -1, (0, 255, 0), 3)
        plt.imshow(image)
        return selected_contours
    return color_filtering(image)

png_path = r"C:\Users\mete\Documents\Github-Rep\medical-tracking\data\png"
valid_images = [".png"]
sort_algorithm = Sort()
debug_tracks = []
for f in os.listdir(png_path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img = cv2.imread(os.path.join(png_path,f))
    dets = detector(img)
    debug_tracks.append(sort_algorithm.update(dets))






