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

def fit(cfg,
        sam_model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        model_save_path):
    """
    Function to fit model
    """
    
    best_valid_dice = 0
    
    device = cfg.base.gpu_id
    num_epochs = cfg.train.num_epochs
    
    for epoch in range(num_epochs):
        sam_model.train()
        
        epoch_loss = 0
        valid_dice = 0
        
        print(f"Epoch #{epoch+1}/{num_epochs}")
        for step, batch in enumerate(tqdm(train_loader, desc='Model training', unit='batch', leave=True)):

            """
            Load precomputed image embeddings to ease training process
            We also load mask labels and bounding boxes directly computed from ground truth masks
            """
            image, true_mask, boxes = batch['image'], batch['mask'], batch['bboxes']
            sam_model = sam_model.to(f"cuda:{device}")
            image = image.to(f"cuda:{device}")
            true_mask = true_mask.to(f"cuda:{device}")

            """
            We freeze image encoder & prompt encoder, only finetune mask decoder
            """
            with torch.no_grad():
                """
                Compute image embeddings from a batch of images with SAM's frozen encoder
                """
                encoder = torch.nn.DataParallel(sam_model.image_encoder, device_ids=[3, 2, 1, 0], output_device=device)
                encoder = encoder.to(f"cuda:{encoder.device_ids[0]}")
                sam_model = sam_model.to(f"cuda:{encoder.device_ids[0]}")
                image = image.to(f"cuda:{encoder.device_ids[0]}")
                image = sam_model.preprocess(image[:, :, :])
                image_embedding = encoder(image)

                """
                Get bounding boxes to make segmentation prediction
                We follow the work by Jun Ma & Bo Wang in Segment Anything in Medical Images (2023)
                to get bounding boxes from the masks as the boxes prompt for SAM
                """
                box_np = boxes.numpy()
                sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
                box = sam_trans.apply_boxes(box_np, (true_mask.shape[-2], true_mask.shape[-1]))
                box_torch = torch.as_tensor(box, dtype=torch.float, device=f"cuda:{device}")
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :] # (B, 1, 4)
                
                """
                Encode box prompts information with SAM's frozen prompt encoder
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
            We now finetune mask decoder
            """
            sam_model = sam_model.to(f"cuda:{device}")
            predicted_mask, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(f"cuda:{device}"), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            ) # -> (B, 1, 256, 256)

            predicted_mask = predicted_mask.to(f"cuda:{device}")
            true_mask = true_mask.to(f"cuda:{device}")
            loss = criterion(predicted_mask, true_mask)
            
            """
            Upgrade model's params
            """
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        
            clip_value = 1 # Clip gradient
            torch.nn.utils.clip_grad_norm_(sam_model.mask_decoder.parameters(), clip_value)
        
            optimizer.step()
            epoch_loss += loss.item()
        
        """
        Validation step with Dice as the metric
        """
        with torch.no_grad():
            valid_dice = eval_dice(sam_model,
                                valid_loader,
                                device=device)
    
        epoch_loss /= ((step + 1) * len(train_loader))
        print(f'Loss: {epoch_loss}\n---')
        
        """
        Save best model
        """
        if best_valid_dice < valid_dice:
            best_valid_dice = valid_dice
            torch.save(sam_model.state_dict(), join(model_save_path, f'{cfg.base.best_valid_model_checkpoint}{cfg.base.random_seed}.pth'))
        
        print(f"Valid dice: {valid_dice*100}")
        print('=======================================')
            
    print(f"Best valid dice: {best_valid_dice*100}")

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

def medsam_2d(yml_args, cfg):
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
    model_save_path = join("./work_dir", 'SAM-ViT-B')
    os.makedirs(model_save_path, exist_ok=True)

    print(f"Fine-tuned SAM in {cfg.base.dataset_name} with {cfg.train.optimizer}, LR = {cfg.train.learning_rate}")

    """
    Load SAM with its original checkpoint 
    """
    sam_model = sam_model_registry["vit_b"](checkpoint=cfg.base.original_checkpoint)

    """
    Load precomputed embeddings
    """    
    train_loader, valid_loader, test_loader, _, _ = sam_dataloader(cfg)

    """
    Optimizer & learning rate scheduler config
    """
    if cfg.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(sam_model.mask_decoder.parameters(),
                                    lr=float(cfg.train.learning_rate),
                                    momentum=0.9)
    elif cfg.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), 
                                     lr=float(cfg.train.learning_rate),
                                     weight_decay=0,
                                     amsgrad=True)
    elif cfg.train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(sam_model.mask_decoder.parameters(), 
                                     lr=float(cfg.train.learning_rate),
                                     weight_decay=0)
    else:
        raise NotImplementedError(f"Optimizer {cfg.train.optimizer} is not set up yet")
    
    """
    Loss function
    In this work, we use a combination of Dice and Cross Entropy Loss to measure SAM's loss values
    """
    criterion = monai.losses.DiceCELoss(sigmoid=True, 
                                        squared_pred=True, 
                                        reduction='mean')

    """
    Train model
    """
    if not yml_args.use_test_mode:
        fit(cfg,
            sam_model=sam_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            criterion=criterion,
            model_save_path=model_save_path)

    """
    Test model
    """
    with torch.no_grad():
        sam_model_test_dice = sam_model_registry["vit_b"](checkpoint=join(model_save_path, f'{cfg.base.best_valid_model_checkpoint}{cfg.base.random_seed}.pth'))
        
        sam_model_test_dice.eval()
        test_dice_score = eval_dice(sam_model_test_dice,
                                    test_loader,
                                    device=cfg.base.gpu_id)
        print(f"Test dice score after training with {cfg.train.optimizer}(lr = {cfg.train.learning_rate}): {test_dice_score*100}")