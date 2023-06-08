import logging
import os
import numpy as np
import torch
import cv2
from skimage.transform import resize
from torch.utils.data import Dataset

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    bbox = []
    
    if len(mask.shape) == 2: #(H, W)
        if np.all(mask == 0):
            y_indices, x_indices = np.random.normal(0, 1024, 2)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(1024, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(1024, y_max + np.random.randint(0, 20))
        else:
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        return np.array([x_min, y_min, x_max, y_max])
    
    for i in range(mask.shape[0]):
        if np.all(mask[i] == 0):
            y_indices, x_indices = np.random.normal(0, 1024, 2)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(1024, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(1024, y_max + np.random.randint(0, 20))
        else:
            y_indices, x_indices = np.where(mask[i] > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = mask[i].shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        bbox.append([x_min, y_min, x_max, y_max])
    return np.array(bbox)


class SegmentationDataset(Dataset):
    def __init__(self, name_dataset: str, images_dir: str, masks_dir: str, scale: float = 1.0):
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
                     (1024, 1024), 
                     order=0, 
                     preserve_range=True, 
                     anti_aliasing=False).astype('uint8')
        img = np.asarray(img)
        if not is_mask:
            img = ((img - img.min()) * (1/(0.01 + img.max() - img.min()) * 255)).astype('uint8')
        if is_mask:
            img = resize(img, 
                         (256, 256), 
                         order=0, 
                         preserve_range=True, 
                         anti_aliasing=False).astype('uint8')
        return img

    @classmethod
    def load(self, filename, is_mask=False):
        if is_mask:
            return cv2.imread(filename, 0)
        else:
            return cv2.imread(filename)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        name = self.ids[idx]
        
        if self.name_dataset == "idrid":
            mask_file = os.path.join(self.masks_dir, name)
        elif self.name_dataset == "fgadr":
            mask_file = os.path.join(self.masks_dir, name)
        elif self.name_dataset == "lits":
            mask_file = os.path.join(self.masks_dir, name).replace("volume", "segmentation")
        elif self.name_dataset == "isiconlytrain":
            mask_file = os.path.join(self.masks_dir, name)
            mask_file =  mask_file + "_segmentation.png"
        elif self.name_dataset == "buidnewprocess":
            mask_file = os.path.join(self.masks_dir, name)
        elif self.name_dataset == "kvasir":
            mask_file = os.path.join(self.masks_dir, name)
        else:
            mask_file = os.path.join(self.masks_dir, name)

        img_file = os.path.join(self.images_dir, name)  

        mask = self.load(mask_file, is_mask=True)
        img = self.load(img_file, is_mask=False)
        
        assert mask is not None, mask_file
        assert img is not None, img_file

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        if self.name_dataset == "idrid":
            mask[mask < 50] = 0
            mask[mask >= 50] = 1
        elif self.name_dataset == "fgadr":
            mask[mask < 50] = 0
            mask[mask >= 50] = 1
        elif self.name_dataset in ["kvasir", "buidnewprocess"]:
            mask[mask < 50] = 0
            mask[mask > 200] = 1
        elif self.name_dataset == "isiconlytrain":
            mask[mask > 1] = 1 
        else:
            mask[mask>0] = 1

        bboxes = get_bbox_from_mask(mask)

        data = {
            'image': torch.as_tensor(img.copy()).permute(2, 0, 1).float().contiguous(),
            'mask': torch.tensor(mask[None, :,:]).long(),
            'bboxes' : torch.tensor(bboxes).float(),
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
            masks = []
            bboxes = []
            for idx in items:
                d = self.__getitem__(idx)
                images.append(d['image'])
                masks.append(d['mask'])
                bboxes.append(d['bboxes'])
            # store third dimension in image channels
            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            bboxes = torch.stack(bboxes, dim=0)
            _3d_data = {'image': images, 'mask': masks, 'bboxes': bboxes}
            yield _3d_data

class SegmentationDataset_aug(Dataset):
    def __init__(self, name_dataset: str, images_dir: str, masks_dir: str, scale: float = 1.0, transform=True):

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform = transform
        self.name_dataset = name_dataset
        self.ids = os.listdir(images_dir)
        if len(self.ids) == 0:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(self, img, scale, is_mask, transform):
        img = resize(img, 
                     (1024, 1024), 
                     order=0, 
                     preserve_range=True, 
                     anti_aliasing=False).astype('uint8')
        img = np.asarray(img)
        if (not is_mask) and transform:
            img = ((img - img.min()) * (1/(0.01 + img.max() - img.min()) * 255)).astype('uint8')
        if is_mask:
            img = resize(img, 
                         (256, 256), 
                         order=0, 
                         preserve_range=True, 
                         anti_aliasing=False).astype('uint8')
        return img

    @classmethod
    def preprocess_non_expand(self, img, scale, is_mask, transform):
        img = resize(img, 
                     (1024, 1024), 
                     order=0, 
                     preserve_range=True, 
                     anti_aliasing=False).astype('uint8')
        img = np.asarray(img)
        if (not is_mask) and transform:
            img = ((img - img.min()) * (1/(0.01 + img.max() - img.min()) * 255)).astype('uint8')
        if is_mask:
            img = resize(img, 
                         (256, 256), 
                         order=0, 
                         preserve_range=True, 
                         anti_aliasing=False).astype('uint8')
        return img

    @classmethod
    def load(self, filename, is_mask=False):
        if is_mask:
            return cv2.imread(filename, 0)
        else:
            return cv2.imread(filename)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        name = self.ids[idx]
        if self.name_dataset == "idrid":
            mask_file = os.path.join(self.masks_dir, name).split('.npy')[0] + '.jpg'
        elif self.name_dataset == "fgadr":
            mask_file = os.path.join(self.masks_dir, name).split('.npy')[0] + '.png'

        img_file = os.path.join(self.images_dir, name)

        mask = self.load(mask_file, is_mask=True)
        img = self.load(img_file, is_mask=False)

        assert mask is not None, mask_file
        assert img is not None, img_file
        
        img = self.preprocess_non_expand(img, self.scale, False, self.transform)
        mask = self.preprocess(mask, self.scale, True, self.transform)
        
        if self.name_dataset == "idrid":
            mask[mask < 50] = 0
            mask[mask >= 50] = 1
        elif self.name_dataset == "fgadr":
            mask[mask < 50] = 0
            mask[mask >= 50] = 1
        else:
            mask[mask>0]=1

        bboxes = get_bbox_from_mask(mask)

        data = {
            'image': torch.as_tensor(img.copy()).permute(2, 0, 1).float().contiguous(),
            'mask': torch.as_tensor(mask.copy().astype(int)).long().contiguous(),
            'bboxes' : torch.tensor(bboxes).float(),
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
            masks = []
            bboxes = []
            for idx in items:
                d = self.__getitem__(idx)
                images.append(d['image'])
                masks.append(d['mask'])
                bboxes.append(d['bboxes'])
            # store third dimension in image channels
            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            bboxes = torch.stack(bboxes, dim=0)
            _3d_data = {'image': images, 'mask': masks, 'bboxes': bboxes}
            yield _3d_data