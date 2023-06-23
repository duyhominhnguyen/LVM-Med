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

def split_data(datasetname):
    dir_image = "./dataset_demo/datasetname/images"
    dir_label = "./dataset_demo/datasetname/labels"

    dir_image = dir_image.replace("datasetname", datasetname)
    dir_label = dir_label.replace("datasetname", datasetname)

    path = "./dataset_demo"
    filesplit = "./files_split"

    file_data = os.path.join(filesplit, datasetname+".json")
    path_data = os.path.join(path, datasetname)


    with open(file_data, 'r') as openfile:
        json_object = json.load(openfile)

    for typ in json_object.keys():
        print(typ)
        file_type = json_object[typ]
        path_type = os.path.join(path_data, typ)
        if os.path.exists(path_type):
            shutil.rmtree(path_type)
        os.mkdir(path_type)

        for img in file_type:
            if not "labels" in typ:
                shutil.copy(os.path.join(dir_image, img), os.path.join(path_type, img.replace(" ", "")))
            else:
                shutil.copy(os.path.join(dir_label, img), os.path.join(path_type, img.replace(" ", "")))
                
def save_fileLabel(datasetname):
    file_image = "./dataset_demo/datasetname/type"
    file_label = "./dataset_demo/datasetname/type_labels"
    path = "./dataset_demo/datasetname"
#     if datasetname == "BUID":
#         B = ['train', 'valid']
#     else:
    B = ['train']
    file_image = file_image.replace("datasetname", datasetname)
    file_label = file_label.replace("datasetname", datasetname)
    path = path.replace("datasetname", datasetname)
    with open(os.path.join(path, 'have_label.txt'), 'w') as F:
        for j in B:
            dir_label = file_label.replace("type", j)
            dir_train = file_image.replace("type", j)
            for i in os.listdir(dir_label):
                file = os.path.join(dir_label, i)
                img = np.array(Image.open(file))
                img[img < 50] = 0
                img[img > 200] = 1
                values, counts = np.unique(img, return_counts = True)
                if (len(values) == 2):
                        a = i.replace("_segmentation.png", ".jpg")
                        F.write(os.path.join(dir_train, a))
                        F.write(" ")
                        F.write(file)
                        F.write(" ")
                        F.write("[0, 1]")
                        F.write("\n")
    
    with open(os.path.join(path, 'non_label.txt'), 'w') as F:
        for j in B:
            dir_label = file_label.replace("type", j)
            dir_train = file_image.replace("type", j)
            for i in os.listdir(dir_label):
                file = os.path.join(dir_label, i)
                img = np.array(Image.open(file))
                img[img < 50] = 0
                img[img > 200] = 1
                values, counts = np.unique(img, return_counts = True)

                if (len(values) == 1):
                        a = i.replace("_segmentation.png", ".jpg")
                        F.write(os.path.join(dir_train, a))
                        F.write(" ")
                        F.write(file)
                        F.write(" ")
                        F.write("[0]")
                        F.write("\n")

def save_fileLabel_3D(datasetname):
    file_image = "./dataset_demo/datasetname/type"
    file_label = "./dataset_demo/datasetname/type_labels"
    path = "./dataset_demo/datasetname"
#     if datasetname == "BUID":
#         B = ['train', 'valid']
#     else:
    B = ['train']
    file_image = file_image.replace("datasetname", datasetname)
    file_label = file_label.replace("datasetname", datasetname)
    path = path.replace("datasetname", datasetname)
    with open(os.path.join(path, 'have_label.txt'), 'w') as F:
        for j in B:
            dir_label = file_label.replace("type", j)
            dir_train = file_image.replace("type", j)
            for i in os.listdir(dir_label):
                file = os.path.join(dir_label, i)
                img = np.array(Image.open(file))
                values, counts = np.unique(img, return_counts = True)
                if 60 in values:
                        a = i.replace("label", "image")
                        F.write(os.path.join(dir_train, a))
                        F.write(" ")
                        F.write(file)
                        F.write(" ")
                        F.write("[0, 2]")
                        F.write("\n")
    
    with open(os.path.join(path, 'non_label.txt'), 'w') as F:
        for j in B:
            dir_label = file_label.replace("type", j)
            dir_train = file_image.replace("type", j)
            for i in os.listdir(dir_label):
                file = os.path.join(dir_label, i)
                img = np.array(Image.open(file))
                values, counts = np.unique(img, return_counts = True)

                if not 60 in values:
                        a = i.replace("label", "image")
                        F.write(os.path.join(dir_train, a))
                        F.write(" ")
                        F.write(file)
                        F.write(" ")
                        F.write("[0]")
                        F.write("\n")