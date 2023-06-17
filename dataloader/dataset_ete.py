import logging
import os
import numpy as np
import torch
import cv2
from skimage.transform import resize
from torch.utils.data import Dataset



class SegmentationDataset_train(Dataset):
    def __init__(self, nonlabel_path: str, havelabel_path: str, dataset: str, scale = (224, 224)):
        self.nonlabel_path = nonlabel_path
        self.havelabel_path = havelabel_path
        self.name_dataset = dataset
        self.scale = scale

        with open(self.nonlabel_path, 'r') as nlf:
            lines = nlf.readlines()
            non_label_lines = [line.strip().split(' ')[:2] for line in lines]
        
        with open(self.havelabel_path, 'r') as hlf:
            lines = hlf.readlines()
            have_label_lines = [line.strip().split(' ')[:2] for line in lines]

        if len(non_label_lines) == 0:
            self.ids = np.array(have_label_lines, dtype= object)
        else:
            choose_non_lable_lines = np.random.choice(len(non_label_lines), size = len(have_label_lines))
            non_label_lines = np.array(non_label_lines, dtype= object)
            have_label_lines = np.array(have_label_lines, dtype= object)
            self.ids = np.concatenate([non_label_lines[choose_non_lable_lines], have_label_lines], axis= 0)
        # self.ids = os.listdir(images_dir) #[splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.') and image_type in file]
        # print(len(self.ids))
        # if datasetname == "las_mri":
        #     self.ids = [f for f in self.ids if image_type in f]
        if len(self.ids) == 0:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.cache = {}

    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def preprocess(self, img, scale, is_mask):
        img = resize(img, 
                     (scale[0], scale[0]), 
                     order=0, 
                     preserve_range=True, 
                     anti_aliasing=False).astype('uint8')
        img = np.asarray(img)
        if not is_mask:
            img = ((img - img.min()) * (1/(0.01 + img.max() - img.min()) * 255)).astype('uint8')
            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2) #(1, 224, 224)
        if is_mask:
            img = resize(img, 
                         (scale[1], scale[1]), 
                         order=0, 
                         preserve_range=True, 
                         anti_aliasing=False).astype('uint8')
        return img

    @classmethod
    def load(self, filename, name_dataset, is_mask=False):
        if name_dataset.startswith("las"):
            if is_mask:
                return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(filename, 0)
                return img
        else: 
            if is_mask:
                return cv2.imread(filename, 0)
            else:
                return cv2.imread(filename)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img_file = self.ids[idx][0]
        mask_file = self.ids[idx][1]
        # print(img_file)
        #start = time.time()
        mask = self.load(mask_file, self.name_dataset, is_mask=True)
        img = self.load(img_file, self.name_dataset, is_mask=False)
        
        assert mask is not None, mask_file
        assert img is not None, img_file


        if self.name_dataset in ["kvasir", "buidnewprocess"]:
            mask[mask < 50] = 0
            mask[mask > 200] = 1
        elif self.name_dataset == "isiconlytrain":
            mask[mask > 1] = 1 
        elif self.name_dataset.startswith("las"):
            mask[mask == 30] = 1
            mask[mask == 60] = 2 # main predict
            mask[mask == 90] = 3
            mask[mask == 120] = 4
            mask[mask == 150] = 5
            mask[mask == 180] = 6
            mask[mask == 210] = 7
            mask[mask > 7] = 0
        else:
            mask[mask>0] = 1

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        data = {
            'image': torch.as_tensor(img.copy()).permute(2, 0, 1).float().contiguous(),
            'mask_ete': torch.as_tensor(mask.copy().astype(int)).long().contiguous(),
            'mask_file' : mask_file,
            'img_file' : img_file
        }
        self.cache[idx] = data
        return data

    def get_3d_iter(self):
        from itertools import groupby
        keyf = lambda idx : self.ids[idx].split("_frame_")[0]
        sorted_ids = sorted(range(len(self.ids)), key=lambda i : self.ids[i])
        for _, items in groupby(sorted_ids, key=keyf):
            images = []
            masks_ete = []
            for idx in items:
                d = self.__getitem__(idx)
                images.append(d['image'])
                masks_ete.append(d['mask_ete'])
            # store third dimension in image channels
            images = torch.stack(images, dim=0)
            masks_ete = torch.stack(masks_ete, dim=0)
            _3d_data = {'image': images, 'mask_ete': masks_ete}
            yield _3d_data
            
            
class SegmentationDataset(Dataset):
    def __init__(self, name_dataset: str, images_dir: str, masks_dir: str, scale = (1024, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.name_dataset = name_dataset
        self.ids = os.listdir(images_dir) 
        if len(self.ids) == 0:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(self, img, scale, is_mask):
        img = resize(img, 
                     (scale[0], scale[0]), 
                     order=0, 
                     preserve_range=True, 
                     anti_aliasing=False).astype('uint8')
        img = np.asarray(img)
        if not is_mask:
            img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
            if len(img.shape) != 3:
                img = np.expand_dims(img, axis=2) #(1, 224, 224)
        if is_mask:
            img = resize(img, 
                         (scale[1], scale[1]), 
                         order=0, 
                         preserve_range=True, 
                         anti_aliasing=False).astype('uint8')
        return img

    @classmethod
    def load(self, filename, name_dataset, is_mask=False):
        if name_dataset.startswith("las"):
            if is_mask:
                return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(filename, 0)
                return img
        else: 
            if is_mask:
                return cv2.imread(filename, 0)
            else:
                return cv2.imread(filename)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        name = self.ids[idx]
        
        if self.name_dataset == "isiconlytrain":
            mask_file = os.path.join(self.masks_dir, name).split(".jpg")[0]
            mask_file =  mask_file + "_segmentation.png"
        elif self.name_dataset == "buidnewprocess":
            mask_file = os.path.join(self.masks_dir, name)
        elif self.name_dataset == "kvasir":
            mask_file = os.path.join(self.masks_dir, name)
        elif self.name_dataset == "drive":
            mask_file = os.path.join(self.masks_dir, name).replace("training", "manual1")
        elif self.name_dataset == "bts":
            mask_file = os.path.join(self.masks_dir, name).replace(self.image_type, "_seg_")
        elif self.name_dataset in ["las_mri", "las_ct"]:
            mask_file = os.path.join(self.masks_dir, name).replace("image", "label")
        else:
            mask_file = os.path.join(self.masks_dir, name)

        img_file = os.path.join(self.images_dir, name)  

        mask = self.load(mask_file, self.name_dataset, is_mask=True)
        img = self.load(img_file, self.name_dataset, is_mask=False)
        
        assert mask is not None, mask_file
        assert img is not None, img_file

       

        if self.name_dataset in ["kvasir", "buidnewprocess"]:
            mask[mask < 50] = 0
            mask[mask > 200] = 1
        elif self.name_dataset == "isiconlytrain":
            mask[mask > 1] = 1 
        elif self.name_dataset.startswith("las"):
            mask[mask == 30] = 1
            mask[mask == 60] = 2 # main predict
            mask[mask == 90] = 3
            mask[mask == 120] = 4
            mask[mask == 150] = 5
            mask[mask == 180] = 6
            mask[mask == 210] = 7
            mask[mask > 7] = 0
        else:
            mask[mask>0] = 1

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        data = {
            'image': torch.as_tensor(img.copy()).permute(2, 0, 1).float().contiguous(),
            'mask_ete': torch.as_tensor(mask.copy().astype(int)).long().contiguous(),
            'mask_file' : mask_file,
            'img_file' : img_file
        }
        self.cache[idx] = data
        return data

    def get_3d_iter(self):
        from itertools import groupby
        keyf = lambda idx : self.ids[idx].split("_frame_")[0]
        sorted_ids = sorted(range(len(self.ids)), key=lambda i : self.ids[i])
        for _, items in groupby(sorted_ids, key=keyf):
            images = []
            masks_ete = []
            for idx in items:
                d = self.__getitem__(idx)
                images.append(d['image'])
                masks_ete.append(d['mask_ete'])
            # store third dimension in image channels
            images = torch.stack(images, dim=0)
            masks_ete = torch.stack(masks_ete, dim=0)
            _3d_data = {'image': images, 'mask_ete': masks_ete}
            yield _3d_data

