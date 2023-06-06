from torch.utils.data import (
    DataLoader
)
import torch
from os import listdir
from os.path import splitext
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader.dataset import (
    SegmentationDataset,
    SegmentationDataset_aug,
)
from dataloader.FLARE_LITS_dataset import (
    FLARE_LiTS_SegmentationDatasetTrain,
    FLARE_LiTS_SegmentationDatasetTest
)
from dataloader.precomputed_dataset import (
    PrecomputedSegmentationDataset,
    FLARE_LITS_PrecomputedSegmentationDataset
)

def get_dataloader(args,
                    dataset_name: str,
                   img_shape: int, 
                   num_workers: int,
                   train_batch_size: int,
                   val_batch_size: int,
                   test_batch_size: int):
    if dataset_name == "idrid":
        if args.dataset_id < 0 or args.dataset_id > 3:
            raise NotImplementedError(f"[Error] Dataset id is not existed!")

        seg_class = ["Haemorrhages", "Hard Exudates", "Microaneurysms", "Soft Exudates"]
        
        train_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/train')
        val_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/valid')
        test_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/test')

        train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/train_labels/' + seg_class[args.dataset_id])
        val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/valid_labels/' + seg_class[args.dataset_id])
        test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/test_labels/' + seg_class[args.dataset_id])
    elif dataset_name == "fgadr":
        if args.dataset_id < 0 or args.dataset_id > 3:
            raise NotImplementedError(f"[Error] Dataset id is not existed!")

        seg_class = ["Haemorrhages", "Hard Exudates", "Microaneurysms", "Soft Exudates"]
        
        train_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/train/')
        val_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/valid/')
        test_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/test/')

        train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/train_labels/' + seg_class[args.dataset_id])
        val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/valid_labels/' + seg_class[args.dataset_id])
        test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/test_labels/' + seg_class[args.dataset_id])
    elif dataset_name == "flare":
        train_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/train')
        val_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/valid')
        test_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/test')
        
        train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/train_labels')
        val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/valid_labels')
        test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/test_labels')
    elif dataset_name == "lits":
        train_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/train')
        val_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/valid')
        test_dir_img = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/test')
        
        train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/train_labels')
        val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/valid_labels')
        test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/test_labels')
    else:
        raise NameError(f'Dataset {dataset_name} is not currently supported!')

    if dataset_name in ["fgadr", "idrid"]:
        train_dataset = SegmentationDataset_aug(dataset_name,
                                            train_dir_img,
                                            train_dir_mask,
                                            scale=img_shape[1],
                                            transform=True)
        val_dataset = SegmentationDataset(dataset_name,
                                        val_dir_img, 
                                        val_dir_mask, 
                                        scale=img_shape[1])
        test_dataset = SegmentationDataset(dataset_name,
                                            test_dir_img, 
                                        test_dir_mask, 
                                        scale=img_shape[1])
    
    elif dataset_name in ["flare", "lits"]:
        train_dataset = FLARE_LiTS_SegmentationDatasetTest(
            train_dir_img,
            train_dir_mask,
            img_shape[1],
            datasetname=dataset_name
        )
        val_dataset = FLARE_LiTS_SegmentationDatasetTest(
            val_dir_img,
            val_dir_mask,
            img_shape[1],
            datasetname=dataset_name
        )
        test_dataset = FLARE_LiTS_SegmentationDatasetTest(
            test_dir_img,
            test_dir_mask,
            img_shape[1],
            datasetname=dataset_name
        )
        # raise NotImplementedError(f"[Error] Dataset {dataset_name} is not either wrong or not yet implemented!")
    else:
        raise NotImplementedError(f"[Error] Dataset {dataset_name} is not either wrong or not yet implemented!")
        
    loader_args = dict(num_workers=num_workers, 
                       pin_memory=torch.cuda.is_available())
    
    train_loader = DataLoader(train_dataset, 
                              shuffle=True, 
                              batch_size=train_batch_size,
                              multiprocessing_context="fork",
                              **loader_args)
    val_loader = DataLoader(val_dataset, 
                            shuffle=False, 
                            drop_last=True, 
                            batch_size=val_batch_size,  
                            multiprocessing_context="fork",
                            **loader_args)
    test_loader = DataLoader(test_dataset, 
                             shuffle=False, 
                             batch_size=test_batch_size,
                             drop_last=True, 
                             multiprocessing_context="fork",
                             **loader_args)
    
    return train_loader, val_loader, test_loader

def get_precomputed_dataloader(args,
                            img_shape: int, 
                            num_workers: int,
                            train_batch_size: int,
                            val_batch_size: int,
                            test_batch_size: int):
    if args.task == 'sam':
        if args.dataset_name == "idrid":
            if args.dataset_id < 0 or args.dataset_id > 3:
                raise NotImplementedError(f"[Error] Dataset id is not existed!")
                
            seg_class = ["Haemorrhages", "Hard Exudates", "Microaneurysms", "Soft Exudates"]
            
            train_dir_img = Path(f'/home/tannp/ICML_workshop/dataloader/sam_embeddings/idrid/train')
            val_dir_img = Path(f'/home/tannp/ICML_workshop/dataloader/sam_embeddings/idrid/valid')
            test_dir_img = Path(f'/home/tannp/ICML_workshop/dataloader/sam_embeddings/idrid/test')

            train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/train_labels/' + seg_class[args.dataset_id])
            val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/valid_labels/' + seg_class[args.dataset_id])
            test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/IDRiD_Segmentation/test_labels/' + seg_class[args.dataset_id])
        elif args.dataset_name == "fgadr":
            seg_class = ["Haemorrhages", "Hard Exudates", "Microaneurysms", "Soft Exudates"]
            train_dir_img = Path(f'/home/tannp/ICML_workshop/dataloader/sam_embeddings/fgadr/train')
            val_dir_img = Path(f'/home/tannp/ICML_workshop/dataloader/sam_embeddings/fgadr/valid')
            test_dir_img = Path(f'/home/tannp/ICML_workshop/dataloader/sam_embeddings/fgadr/test')

            train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/train_labels/' + seg_class[args.dataset_id])
            val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/valid_labels/' + seg_class[args.dataset_id])
            test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FGADR_Segmentation/test_labels/' + seg_class[args.dataset_id])
        elif args.dataset_name == "flare":
            train_dir_img = Path('/home/tannp/ICML_workshop/dataloader/sam_embeddings/flare/train')
            val_dir_img = Path('/home/tannp/ICML_workshop/dataloader/sam_embeddings/flare/valid')
            test_dir_img = Path('/home/tannp/ICML_workshop/dataloader/sam_embeddings/flare/test')
            
            train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/train_labels')
            val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/valid_labels')
            test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/FLARE22Train/test_labels')

            nonlabel_path = Path('/home/tannp/ICML_workshop/dataloader/sam_embeddings/flare/non_label.txt')
            havelabel_path = Path('/home/tannp/ICML_workshop/dataloader/sam_embeddings/flare/have_label.txt')
        
        elif args.dataset_name == "lits":
            train_dir_img = Path('/raid/hnguyen/SAM/tannp/sam_embeddings/lits/train')
            val_dir_img = Path('/raid/hnguyen/SAM/tannp/sam_embeddings/lits/valid')
            test_dir_img = Path('/raid/hnguyen/SAM/tannp/sam_embeddings/lits/test')
            
            train_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/train_labels')
            val_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/valid_labels')
            test_dir_mask = Path('/home/tannp/ICML_workshop/dataloader/Segmentation_dataset/LITS/test_labels')

            nonlabel_path = Path('/raid/hnguyen/SAM/tannp/sam_embeddings/lits/non_label.txt')
            havelabel_path = Path('/raid/hnguyen/SAM/tannp/sam_embeddings/lits/have_label.txt')
        else:
            raise NameError(f'Dataset {args.dataset_name} is not currently supported!')
    else:
        raise NameError(f'Task {args.task} is not currently supported!')

    loader_args = dict(num_workers=num_workers, 
                    #    pin_memory=torch.cuda.is_available())
                        pin_memory=False)

    if args.dataset_name in ["idrid", "fgadr"]:
        train_dataset = PrecomputedSegmentationDataset(args.dataset_name,
                                                    train_dir_img,
                                                    train_dir_mask,
                                                    scale=img_shape[1])
        
    elif args.dataset_name in ["flare", "lits"]:
        train_dataset = FLARE_LITS_PrecomputedSegmentationDataset(nonlabel_path,
                                                                  havelabel_path,
                                                                  args.dataset_name,
                                                                  scale=img_shape[1])
    else:
        raise NotImplementedError(f"[Error] Dataset load method for {args.dataset_name} dataset is either wrong or not yet implemented!")

    val_dataset = PrecomputedSegmentationDataset(args.dataset_name,
                                                    val_dir_img, 
                                                    val_dir_mask, 
                                                    scale=img_shape[1])
    test_dataset = PrecomputedSegmentationDataset(args.dataset_name,
                                                test_dir_img, 
                                                test_dir_mask, 
                                                scale=img_shape[1])
    train_loader = DataLoader(train_dataset, 
                              shuffle=True, 
                              batch_size=train_batch_size,
                              multiprocessing_context="fork",
                              **loader_args)
    val_loader = DataLoader(val_dataset, 
                            shuffle=False, 
                            drop_last=True, 
                            batch_size=val_batch_size,  
                            multiprocessing_context="fork",
                            **loader_args)
    test_loader = DataLoader(test_dataset, 
                             shuffle=False, 
                             batch_size=test_batch_size,
                             drop_last=True, 
                             multiprocessing_context="fork",
                             **loader_args)
    
    return train_loader, val_loader, test_loader, val_dataset, test_dataset