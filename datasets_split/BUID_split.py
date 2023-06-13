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

def BUID_split():
    dataset_name = "BUID"
    file = "./dataset_demo/BUID"
    if os.path.exists(os.path.join(file, "normal")):
        shutil.rmtree(os.path.join(file, "normal"))

    dir_data = "./dataset_demo/BUID/images"
    dir_label = "./dataset_demo/BUID/labels"

    if os.path.exists(dir_data):
        shutil.rmtree(dir_data)
    os.mkdir(dir_data)

    if os.path.exists(dir_label):
        shutil.rmtree(dir_label)
    os.mkdir(dir_label)

    for i in os.listdir(file):
        if i == "labels" or i == "images" or "label" in i:
            continue
        file_label = os.path.join(file, i)
        for img in os.listdir(file_label):
            img_file = os.path.join(file_label, img)
            if "mask" in img:
                shutil.copy(img_file, os.path.join(dir_label, img))
            else:
                shutil.copy(img_file, os.path.join(dir_data, img))

    file = os.listdir(dir_label)
    label_uni = -1
    check = False
    b = 0
    for i in os.listdir(dir_data):
        for k in range(10):
            if k == 0:
                mask = "_mask"
            else:
                mask = "_mask_" + str(k)
            a = i.replace(".png", mask+".png")

            if a in file:
                b = k
                if not check:
                    label_uni = cv2.imread(os.path.join(dir_label, a))
                    check = True
                else:
                    img = cv2.imread(os.path.join(dir_label, a))
                    label_uni = label_uni + img
                os.remove(os.path.join(dir_label, a))
            else:
                check = False
                break

        #print(i)
        cv2.imwrite(os.path.join(dir_label, i), label_uni)
        label_uni = -1
        check = False
        b = 0

    split_data(dataset_name)
    save_fileLabel(dataset_name)