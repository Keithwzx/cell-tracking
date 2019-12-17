import os
import nibabel as nib

path = r"C:\Users\mete\Documents\Github-Rep\medical-tracking"
output_path = os.path.join(path, "outputs")

for f in os.listdir(output_path):
