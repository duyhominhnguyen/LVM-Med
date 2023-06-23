import math
import os
import random
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import glob
from skimage.io import imsave
from utils.dataset_prepare import split_data, save_fileLabel_3D

# input image is the volume
def __itensity_normalize_one_volume__(image):
    # normalization following Med3D
    top_per = np.percentile(image, 99.5)
    bot_per = np.percentile(image, 0.5)
    image[image > top_per] = top_per
    image[image < bot_per] = bot_per
    image = (image - np.mean(image)) / np.std(image)
    image = image / 10.0
    image[image < 0] = 0.0
    image[image > 1] = 1.0
    return image


def __training_data_process__(data, label): 
    # crop data according net input size
    data = data.get_fdata()
    label = label.get_fdata()

    # normalization datas
    data = __itensity_normalize_one_volume__(data)
    
    # changing label values
    label[label == 205] = 30
    label[label == 420] = 60
    label[label == 500] = 90
    label[label == 550] = 120
    label[label == 600] = 150
    label[label == 820] = 180
    label[label == 850] = 210

    return data, label


def preprocess_vol(img_name, label_name):

    assert os.path.isfile(img_name)
    assert os.path.isfile(label_name)
   
    img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
    assert img is not None
    mask = nibabel.load(label_name)
    assert mask is not None

    img_array, mask_array = __training_data_process__(img, mask)
    assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
        
    return (img_array*255).astype('uint8'), mask_array.astype('uint8')

#if __name__ == '__main__':
def MMWHS_MR_Heart_split():
    dataset_name = "MMWHS_MR_Heart"
    ### Training set
    data_dir = './dataset_demo/MMWHS_MR_Heart/Raw/train/'
    img_fold_list = os.listdir(data_dir)
    dest_dir = './dataset_demo/MMWHS_MR_Heart/train/' # dir for saving train images
    dest_dir_label = './dataset_demo/MMWHS_MR_Heart/train_labels/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not os.path.exists(dest_dir_label):
        os.makedirs(dest_dir_label)

    for vol_name in img_fold_list:
        if 'label' in vol_name:
            continue
        mask_name = os.path.join(data_dir, vol_name).replace('image','label')
        img_flair, mask = preprocess_vol(os.path.join(data_dir, vol_name), mask_name)
        print(img_flair.shape, mask.shape)
        # img_array.shape[2] is the length of depth dimension
        for depth in range(0, img_flair.shape[2]):
            imsave(os.path.join(dest_dir, vol_name.split('.')[0] + '_frame_' + str(depth).zfill(3) + '.png'), img_flair[:, :, depth], check_contrast=False)
            imsave(os.path.join(dest_dir_label, vol_name.replace('image','label').split('.')[0] + '_frame_' + str(depth).zfill(3) + '.png'), mask[:, :, depth], check_contrast=False)

    ### Validation set
    data_dir = './dataset_demo/MMWHS_MR_Heart/Raw/valid/'
    img_fold_list = os.listdir(data_dir)
    dest_dir = './dataset_demo/MMWHS_MR_Heart/valid/'
    dest_dir_label = './dataset_demo/MMWHS_MR_Heart/valid_labels/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not os.path.exists(dest_dir_label):
        os.makedirs(dest_dir_label)

    for vol_name in img_fold_list:
        if 'label' in vol_name:
            continue
        mask_name = os.path.join(data_dir, vol_name).replace('image','label')
        img_flair, mask = preprocess_vol(os.path.join(data_dir, vol_name), mask_name)
        print(img_flair.shape, mask.shape)
        # img_array.shape[2] is the length of depth dimension
        for depth in range(0, img_flair.shape[2]):
            imsave(os.path.join(dest_dir, vol_name.split('.')[0] + '_frame_' + str(depth).zfill(3) + '.png'), img_flair[:, :, depth], check_contrast=False)
            imsave(os.path.join(dest_dir_label, vol_name.replace('image','label').split('.')[0] + '_frame_' + str(depth).zfill(3) + '.png'), mask[:, :, depth], check_contrast=False)

    ### Testing set
    data_dir = './dataset_demo/MMWHS_MR_Heart/Raw/test/'
    img_fold_list = os.listdir(data_dir)
    dest_dir = './dataset_demo/MMWHS_MR_Heart/test/'
    dest_dir_label = './dataset_demo/MMWHS_MR_Heart/test_labels/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not os.path.exists(dest_dir_label):
        os.makedirs(dest_dir_label)

    for vol_name in img_fold_list:
        if 'label' in vol_name:
            continue
        mask_name = os.path.join(data_dir, vol_name).replace('image','label')
        img_flair, mask = preprocess_vol(os.path.join(data_dir, vol_name), mask_name)
        print(img_flair.shape, mask.shape)
        # img_array.shape[2] is the length of depth dimension
        for depth in range(0, img_flair.shape[2]):
            imsave(os.path.join(dest_dir, vol_name.split('.')[0] + '_frame_' + str(depth).zfill(3) + '.png'), img_flair[:, :, depth], check_contrast=False)
            imsave(os.path.join(dest_dir_label, vol_name.replace('image','label').split('.')[0] + '_frame_' + str(depth).zfill(3) + '.png'), mask[:, :, depth], check_contrast=False)
    
    
    save_fileLabel_3D(dataset_name)