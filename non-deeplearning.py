from PIL import Image
import os, os.path
from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from tracking import *
import warnings
warnings.filterwarnings("ignore")


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
                                    cv2.THRESH_BINARY_INV, 31,5)
        kernel = np.ones((8, 8), np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        img_erosion = cv2.erode(closing, np.ones((3, 3), np.uint8), iterations=1)
        plt.imshow(img_erosion)
        contours, hierarchy = cv2.findContours(img_erosion,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected_contours = []
        for contour in contours:
            if len(contour) > 50:
                selected_contours.append(contour)
        # cv2.drawContours(image, selected_contours, -1, (0, 255, 0), 3)
        # plt.imshow(image)
        return selected_contours
    return color_filtering(image)

png_path = r"C:\Users\mete\Documents\Github-Rep\medical-tracking\data\png"
try:
    os.mkdir(os.path.join(png_path,"tracked"))
except:
    pass
valid_images = [".png"]
sort_algorithm = Associator()
debug_tracks = []
for f in os.listdir(png_path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img = cv2.imread(os.path.join(png_path,f))
    dets = detector(img)
    drawed_img = sort_algorithm.display_trackers(img,sort_algorithm.update(dets))
    cv2.imwrite(os.path.join(r"C:\Users\mete\Documents\Github-Rep\medical-tracking\tracking_results",f"{f}"),drawed_img)


trackers = sort_algorithm.trackers
#Sorunlu id=2
temp_tracker = trackers[2]
measurements = temp_tracker.measurements
tracking_results = np.array(temp_tracker.trajectory)
plt.plot(tracking_results[:,0],tracking_results[:,1])






