from PIL import Image
import os, os.path
from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd



def detector(image):
    def color_filtering(image):
        original_image = image
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





