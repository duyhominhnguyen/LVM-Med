import numpy as np
import os 
join = os.path.join
import gc
from tqdm import tqdm
import torch
import monai, random
from dataloader.sam_transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from dataloader.dataloader import sam_dataloader
from utils.SurfaceDice import compute_dice_coefficient

#%% test
def eval_dice(sam_model,
            loader,
            device):
    
    """
    Function to evaluate model (for both validation and testing phase)
    """
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    dice_score = 0.
    for _, batch in enumerate(tqdm(loader, leave=False)):
        """
        Load precomputed embeddings, mask labels and bounding boxes computed directly from ground truth masks
        """
        image, true_mask, boxes = batch['image'], batch['mask'], batch['bboxes']
        image = image.to(f"cuda:{device}")
        true_mask = true_mask.to(f"cuda:{device}", dtype=torch.float32)

        """
        Compute image embeddings
        """
        encoder = torch.nn.DataParallel(sam_model.image_encoder, device_ids=[3, 2, 1, 0], output_device=device)
        encoder = encoder.to(f"cuda:{encoder.device_ids[0]}")
        sam_model = sam_model.to(f"cuda:{encoder.device_ids[0]}")
        image = image.to(f"cuda:{encoder.device_ids[0]}")
        image = sam_model.preprocess(image[:, :, :])
        image_embedding = encoder(image)

        """
        Get bboxes
        """
        box_np = boxes.numpy()
        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, (image_embedding.shape[0], image_embedding.shape[1]))
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        """
        Prompt encoder component
        """
        prompt_encoder = torch.nn.DataParallel(sam_model.prompt_encoder, device_ids=[0,1,2,3], output_device=device)
        prompt_encoder = prompt_encoder.to(f"cuda:{prompt_encoder.device_ids[0]}")
        box_torch = box_torch.to(f"cuda:{prompt_encoder.device_ids[0]}")
        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        """
        Mask decoder component
        """
        sam_model = sam_model.to(f"cuda:{device}")
        mask_segmentation, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(f"cuda:{device}"), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        ) # -> (B, 256, 256)
        
        """
        Transform prediction and evaluate
        """
        true_mask = true_mask.to("cpu")
        medsam_seg_prob = torch.sigmoid(mask_segmentation)
        medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8) # transform from hard masks to soft masks
        dice_score += compute_dice_coefficient(true_mask>0, medsam_seg>0)
    
    return dice_score.cpu().numpy()/len(loader) 

def zero_shot_sam_2d(yml_args, cfg):
    """
    Training warm up
    """
    torch.multiprocessing.set_start_method('spawn')
    
    random.seed(cfg.base.random_seed)
    np.random.seed(cfg.base.random_seed)
    torch.manual_seed(cfg.base.random_seed)
    torch.cuda.manual_seed(cfg.base.random_seed)

    torch.backends.cudnn.deterministic = True

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    """
    General configuration
    """
    img_shape = (3, 1024) # hard settings image shape as 3 x 1024 x 1024

    """
    Load SAM with its original checkpoint 
    """
    sam_model = sam_model_registry["vit_b"](checkpoint=cfg.base.original_checkpoint)

    """
    Load precomputed embeddings
    """    
    _, _, test_loader, _, _ = sam_dataloader(cfg)

    """
    Test model
    """
    with torch.no_grad():
        sam_model.eval()
        test_dice_score = eval_dice(sam_model,
                                    test_loader,
                                    device=cfg.base.gpu_id)
        print(f"Dice score from zero-shot SAM: {test_dice_score*100}")