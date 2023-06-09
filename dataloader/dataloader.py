from torch.utils.data import (
    DataLoader
)
from dataloader.dataset import (
    SegmentationDataset,
    AugmentedSegmentationDataset
)

def embedding_dataloader(cfg):
    loader_args = dict(num_workers=cfg.base.num_workers, 
                        pin_memory=cfg.base.pin_memory)
    if cfg.base.dataset_name in ["buidnewprocess", "kvasir", "isiconlytrain", "drive"]:
        train_dataset = SegmentationDataset(cfg.base.dataset_name,
                                                    cfg.dataloader.train_dir_img,
                                                    cfg.dataloader.train_dir_mask,
                                                    scale=cfg.base.image_shape)
    elif cfg.base.dataset_name in ["bts", "las_mri", "las_ct"]:
        train_dataset = AugmentedSegmentationDataset(cfg.base.dataset_name,
                                                    cfg.dataloader.train_dir_img,
                                                    cfg.dataloader.train_dir_mask,
                                                    scale=cfg.base.image_shape)
    else:
        raise NameError(f"[Error] Dataset {cfg.base.dataset_name} is either in wrong format or not yet implemented!")

    val_dataset = SegmentationDataset(cfg.base.dataset_name,
                                                cfg.dataloader.valid_dir_img, 
                                                cfg.dataloader.valid_dir_mask, 
                                                scale=cfg.base.image_shape)
    test_dataset = SegmentationDataset(cfg.base.dataset_name,
                                                cfg.dataloader.test_dir_img, 
                                                cfg.dataloader.test_dir_mask, 
                                                scale=cfg.base.image_shape)
    train_loader = DataLoader(train_dataset, 
                              shuffle=True, 
                              batch_size=cfg.train.train_batch_size,
                              multiprocessing_context="fork",
                              **loader_args)
    val_loader = DataLoader(val_dataset, 
                            shuffle=False, 
                            drop_last=True, 
                            batch_size=cfg.train.valid_batch_size,  
                            multiprocessing_context="fork",
                            **loader_args)
    test_loader = DataLoader(test_dataset, 
                             shuffle=False, 
                             batch_size=cfg.train.test_batch_size,
                             drop_last=True, 
                             multiprocessing_context="fork",
                             **loader_args)
    
    return train_loader, val_loader, test_loader, val_dataset, test_dataset