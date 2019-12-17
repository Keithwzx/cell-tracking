from PIL import Image
import os, os.path
from matplotlib import pyplot as plt
import cv2
import numpy as np

path = r"C:\Users\mete\Documents\Github-Rep\medical-tracking\data\Fluo-N2DH-GOWT1\01"
png_path = r"C:\Users\mete\Documents\Github-Rep\medical-tracking\data\png"

imgs = []
valid_images = [".png"]
labels = []
for f in os.listdir(png_path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(cv2.imread(os.path.join(png_path,f)))
    labels.append(f"01_{f}")

dictionary_01 = dict(zip(labels,imgs))
image = dictionary_01["01_t091.png"]