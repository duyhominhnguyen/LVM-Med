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

def Kvasir_split():
    dataset_name = "Kvasir"
    file = "./dataset_demo/Kvasir"
    if os.path.exists(os.path.join(file, "masks")):
        os.rename(os.path.join(file, "masks"), os.path.join(file, "labels"))
    split_data(dataset_name)
    save_fileLabel(dataset_name)