import numpy as np
import os 
join = os.path.join
import gc
from tqdm import tqdm
import torch
import monai
from segment_anything import sam_model_registry
from dataloader.sam_transforms import ResizeLongestSide
from dataloader.dataloader import get_precomputed_dataloader
from utils.SurfaceDice import compute_dice_coefficient
import argparse


def fit(args,
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
    device = args.gpu_id
    dataset = args.dataset_name
    num_epochs = args.num_epochs
    
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
            image_embedding, true_mask, boxes = batch['image'], batch['mask'], batch['bboxes']
            sam_model = sam_model.to(f"cuda:{device}")
            image_embedding = image_embedding.to(f"cuda:{device}")

            true_mask.to(f"cuda:{device}")
            
            """
            We freeze image encoder & prompt encoder, only finetune mask decoder
            """
            with torch.no_grad():
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
                Prompt encoder
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
            We now fintune mask decoder
            """
            sam_model = sam_model.to(f"cuda:{device}")
            predicted_mask, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(f"cuda:{device}"), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            predicted_mask = predicted_mask.to(f"cuda:{device}")
            true_mask = true_mask.to(f"cuda:{device}")
            loss = criterion(predicted_mask, true_mask)
            
            """
            Upgrade model's params
            """
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradient
            clip_value = 1 
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
    
        epoch_loss /= (step * len(train_loader))
        print(f'Loss: {epoch_loss}\n---')
        
        """
        Save best model
        """
        if best_valid_dice < valid_dice:
            best_valid_dice = valid_dice
            torch.save(sam_model.state_dict(), join(model_save_path, f'sam_model_best_dice_original_{dataset}_seed{args.random_seed}.pth'))
        
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
        image_embedding, true_mask, boxes = batch['image'], batch['mask'], batch['bboxes']
        true_mask.to(f"cuda:{device}", dtype=torch.float32)

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
        medsam_seg_prob = torch.sigmoid(mask_segmentation)
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8) # transform from hard masks to soft masks
        dice_score += compute_dice_coefficient(true_mask>0, medsam_seg>0)
    
    """
    In this part, I haven't found a more elegant way to do this 
    except for loading evaluation data with batch size = 1
    and take the mean w.r.t. the total number of dataloader length
    """
    return dice_score.cpu().numpy()/len(loader) 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default=f"sam", help='Task to run')
    parser.add_argument('-ds', '--dataset_name', type=str, default=f"fgadr", help='Target domain (dataset) to run')
    parser.add_argument('-ds_id', '--dataset_id', type=int, default=-1, help='Dataset id (organ) to run (Optional, depends on data)')
    parser.add_argument('-opt', '--optimizer', type=str, default=f"sgd", help='Optimizer to run')
    parser.add_argument('-lr', '--learning_rate', type=float, default=f"1", help='learning rate to run')
    parser.add_argument('-bs', '--train_batch_size', type=int, default=20, help='Train batch size')
    parser.add_argument('-n_ep', '--num_epochs', type=int, default=f"5", help='Number of epochs')
    parser.add_argument('-seed', '--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('-gpu', '--gpu_id', type=int, default=3, help='Gpu id to use')
    parser.add_argument('-worker', '--num_workers', type=int, default=40, help='Number of workers')
    parser.add_argument('-test', '--use_test_mode', type=int, default=0, help='Flag for inference only (no train mode)')
    
    args = parser.parse_args()

    """
    Training warm up
    """
    torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    """
    General configuration
    """
    model_save_path = join("./work_dir", 'SAM-ViT-B')
    img_shape = (3, 1024) # hard settings image shape as 3 x 1024 x 1024

    print(f"Fine-tuned SAM in {args.dataset_name} with {args.optimizer}, LR = {args.learning_rate}")
    os.makedirs(model_save_path, exist_ok=True)

    """
    Load SAM with its original checkpoint 
    """
    sam_model = sam_model_registry["vit_b"](checkpoint='work_dir/SAM/sam_vit_b_01ec64.pth')

    """
    Load precomputed embeddings
    """    
    train_loader, valid_loader, test_loader, valid_dataset, test_dataset = get_precomputed_dataloader(args,
                                                                                                    img_shape=img_shape,
                                                                                                    num_workers=args.num_workers,
                                                                                                    train_batch_size=args.train_batch_size,
                                                                                                    val_batch_size=1,
                                                                                                    test_batch_size=1)

    """
    Optimizer & learning rate scheduler config
    """
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(sam_model.mask_decoder.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), 
                                     lr=args.learning_rate,
                                     weight_decay=0,
                                     amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(sam_model.mask_decoder.parameters(), 
                                     lr=args.learning_rate,
                                     weight_decay=0)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} is not set up yet")
    
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
    if not args.use_test_mode:
        fit(args,
            sam_model=sam_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            valid_dataset = valid_dataset,
            optimizer=optimizer,
            criterion=criterion,
            model_save_path=model_save_path)

    """
    Test model
    """
    with torch.no_grad():
        sam_model_test_dice = sam_model_registry["vit_b"](checkpoint=join(model_save_path, f'sam_model_best_dice_original_{args.dataset_name}_seed{args.random_seed}.pth'))
        
        sam_model_test_dice.eval_dice()
        test_dice_score = eval_dice(sam_model_test_dice,
                                test_loader,
                                device=args.gpu_id)
        print(f"Test dice score after training with {args.optimizer}(lr = {args.learning_rate}): {test_dice_score*100}")