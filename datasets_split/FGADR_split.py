import numpy as np
import torch
import shutil
import os
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image
import pickle
from skimage.transform import resize
from utils.dataset_prepare import split_data, save_fileLabel

def FGADR_split():
    pkl_path = './files_split/fgadr_pkl_file.pkl' # change your path here
    path = "./dataset_demo/FGADR"
    f = open(pkl_path, 'rb')
    a = pickle.load(f)
    a_key = a.keys()
    B = ["train", "test"]
    C = ["Training", "Testing"]
    for index, i in enumerate(B):
        print(i)
        print(len(a[i]))
        folder_type = os.path.join(path, i)
        if os.path.exists(folder_type.replace(i, C[index])):
            shutil.rmtree(os.path.join(path, C[index]))
        os.mkdir(os.path.join(path, C[index]))
        for j in a[i]:
            folder_class = os.path.join(folder_type, str(j[1]))
            if not os.path.exists(folder_class.replace(i, C[index])):
                os.mkdir(folder_class.replace(i, C[index]))
            file = j[0].replace("/mnt/sda/haal02-data/FGADR-Seg-Set", "./dataset_demo/FGADR")
            img = cv2.imread(file)
            img = resize(img, (512, 512), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
            #/home/caduser/Foundmed_Experiment/Classification/FGADR/Seg-set/Original_Images/0001_2.png
            name_img = file.split("/")[-1]
            #print(os.path.join(folder_class.replace(i, C[index])))
            cv2.imwrite(os.path.join(folder_class.replace(i, C[index]), name_img), img)