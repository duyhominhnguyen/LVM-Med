import argparse
import logging
import sys
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR

from dataloader.dataset_ete import SegmentationDataset_train, SegmentationDataset
from utils.endtoend import dice_loss
from evaluate import evaluate, evaluate_3d_iou
#from models.segmentation import UNet
import segmentation_models_pytorch as smp
import numpy as np
num_classes = 8
np.random.seed(42)

def train_net(net,
              cfg,
              trial,
              device,
              epochs: int = 30,
              train_batch_size: int = 128,
              val_batch_size: int = 128,
              learning_rate: float = 0.1,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = True,
              out_dir : str= './checkpoint/'):
    # 1. Create dataset
    train_dir_img = Path(cfg.dataloader.train_dir_img)
    train_dir_mask = Path(cfg.dataloader.train_dir_mask)
    val_dir_img = Path(cfg.dataloader.valid_dir_img)
    val_dir_mask = Path(cfg.dataloader.valid_dir_mask)
    test_dir_img = Path(cfg.dataloader.test_dir_img)
    test_dir_mask = Path(cfg.dataloader.test_dir_mask)
    non_label_text = cfg.dataloader.non_label
    have_label_text = cfg.dataloader.have_label

    dir_checkpoint = Path(out_dir)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    
    train_dataset = SegmentationDataset_train(nonlabel_path= non_label_text, havelabel_path= have_label_text, dataset = cfg.base.dataset_name, scale= img_scale)
    
    val_dataset = SegmentationDataset(cfg.base.dataset_name, val_dir_img, val_dir_mask, scale=img_scale)
    
    test_dataset = SegmentationDataset(cfg.base.dataset_name, test_dir_img, test_dir_mask, scale=img_scale)
      
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    

    # 3. Create data loaders
    loader_args = dict(num_workers=10, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, **loader_args)
    import time 
    
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, batch_size=val_batch_size,  **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, train_batch_size=train_batch_size, val_batch_size=val_batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Train batch size:      {train_batch_size}
        Val batch size: {val_batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(cfg.train.beta1, cfg.train.beta2), eps=1e-08, weight_decay=cfg.train.weight_decay)
    if cfg.train.scheduler:
        print("Use scheduler")
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    print(learning_rate)
    # optimizer= optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_value = 0
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask_ete']
                
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                clip_value = 1
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % (n_train // (1 * train_batch_size)) == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_dice_score, val_iou_score = evaluate(net, val_loader, device, 2)
                    val_3d_iou_score = evaluate_3d_iou(net, val_dataset, device, 2)

                    

                    val_score = val_3d_iou_score

                    # scheduler.step(val_dice_score)
                    if (val_score > best_value):
                        best_value = val_score
                        logging.info("New best 3d iou score: {} at epochs {}".format(best_value, epoch+1))
                        torch.save(net.state_dict(), str(dir_checkpoint/'checkpoint_{}_{}_best_{}.pth'.format(cfg.base.dataset_name, cfg.base.original_checkpoint, str(trial))))

                    logging.info('Validation Dice score: {}, IoU score {}, IoU 3d score {}'.format(val_dice_score, val_iou_score, val_3d_iou_score))
                    
                   
        # update learning rate
        if cfg.train.scheduler:
            if (epoch + 1 <= 0):
                scheduler.step()

        # Evaluation the last model
        if epoch + 1 == epochs:
            val_dice_score, val_iou_score = evaluate(net, val_loader, device, 2)
            val_3d_iou_score = evaluate_3d_iou(net, val_dataset, device, 2)
            logging.info('Validation Dice score: {}, IoU score {}, IoU 3d score {}'.format(val_dice_score, val_iou_score, val_3d_iou_score))
            

        if save_checkpoint:
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
            if epoch > 0 and epoch != (epochs % 2 - 1) :
                os.remove( str(dir_checkpoint/'checkpoint_epoch{}.pth'.format(epoch)))
    logging.info("Evalutating on test set")
    logging.info("Loading best model on validation")
    net.load_state_dict(torch.load(str(dir_checkpoint/'checkpoint_{}_{}_best_{}.pth'.format(cfg.base.dataset_name, cfg.base.original_checkpoint, str(trial)))))
    test_dice, test_iou = evaluate(net, test_loader, device, 2)
    test_3d_iou = evaluate_3d_iou(net, test_dataset, device, 2)
    logging.info("Test dice score {}, IoU score {}, 3d IoU {}".format(test_dice, test_iou, test_3d_iou))

    logging.info("Loading model at last epochs %d" %epochs)
    net.load_state_dict(torch.load(str(dir_checkpoint/'checkpoint_epoch{}.pth'.format(epochs))))
    test_dice_last, test_iou_last = evaluate(net, test_loader, device, 2)
    test_3d_iou_last = evaluate_3d_iou(net, test_dataset, device, 2)
    logging.info("Test dice score {}, IoU score {}, 3d IoU {}".format(test_dice_last, test_iou_last, test_3d_iou_last))

    return test_dice, test_iou, test_3d_iou, test_dice_last, test_iou_last, test_3d_iou_last


def eval(cfg, out_dir, net, device, img_scale, trial):
    test_dir_img = Path(cfg.dataloader.test_dir_img)
    test_dir_mask = Path(cfg.dataloader.test_dir_mask)
    test_dataset = SegmentationDataset(name_dataset=cfg.base.dataset_name, images_dir = test_dir_img, masks_dir= test_dir_mask, scale = img_scale)
    loader_args = dict(num_workers=10, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    dir_checkpoint = Path(out_dir)
    
    print("Trial", trial+1)
    logging.info("Evalutating on test set")
    logging.info("Loading best model on validation")
    net.load_state_dict(torch.load(str(dir_checkpoint/'checkpoint_{}_{}_best_{}.pth'.format(cfg.base.dataset_name, cfg.base.original_checkpoint, str(trial)))))
    test_dice, test_iou = evaluate(net, test_loader, device, 2)
    test_3d_iou = evaluate_3d_iou(net, test_dataset, device, 2)
    logging.info("Test dice score {}, IoU score {}, 3d IoU {}".format(test_dice, test_iou, test_3d_iou))
    return test_dice, test_iou, test_3d_iou

def train_3d_R50(yml_args, cfg):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cuda_string = 'cuda:' + cfg.base.gpu_id
    device = torch.device(cuda_string if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    try:
        _2d_dices = []
        _2d_ious = []
        _3d_ious = []
        _2d_dices_last = []
        _2d_ious_last = []
        _3d_ious_last = []

        if not yml_args.use_test_mode:
            for trial in range(5):
                print ("----"*3)
                if cfg.base.original_checkpoint == "scratch":
                    net = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=num_classes)
                else:
                    print ("Using pre-trained models from", cfg.base.original_checkpoint)
                    net = smp.Unet(encoder_name="resnet50", encoder_weights=cfg.base.original_checkpoint ,in_channels=1, classes=num_classes)


                net.to(device=device)

                print("Trial", trial + 1)
                _2d_dice, _2d_iou, _3d_iou, _2d_dice_last, _2d_iou_last, _3d_iou_last = train_net(net=net, cfg=cfg, trial=trial,
                        epochs=cfg.train.num_epochs,
                        train_batch_size=cfg.train.train_batch_size,
                        val_batch_size=cfg.train.valid_batch_size,
                        learning_rate=cfg.train.learning_rate,
                        device=device,
                        img_scale=(cfg.base.image_shape, cfg.base.image_shape),
                        val_percent=10.0 / 100,
                        amp=False,
                        out_dir= cfg.base.best_valid_model_checkpoint)
                _2d_dices.append(_2d_dice.item())
                _2d_ious.append(_2d_iou.item())
                _3d_ious.append(_3d_iou.item())
                _2d_dices_last.append(_2d_dice_last.item())
                _2d_ious_last.append(_2d_iou_last.item())
                _3d_ious_last.append(_3d_iou_last.item())

            print ("Average performance on best valid set")
            print("2d dice {}, mean {}, std {}".format(_2d_dices, np.mean(_2d_dices), np.std(_2d_dices)))
            print("2d iou {}, mean {}, std {}".format(_2d_ious, np.mean(_2d_ious), np.std(_2d_ious)))
            print("3d iou {}, mean {}, std {}".format(_3d_ious, np.mean(_3d_ious), np.std(_3d_ious)))

            print ("Average performance on the last epoch")
            print("2d dice {}, mean {}, std {}".format(_2d_dices_last, np.mean(_2d_dices_last), np.std(_2d_dices_last)))
            print("2d iou {}, mean {}, std {}".format(_2d_ious_last, np.mean(_2d_ious_last), np.std(_2d_ious_last)))
            print("3d iou {}, mean {}, std {}".format(_3d_ious_last, np.mean(_3d_ious_last), np.std(_3d_ious_last)))
        else:
            for trial in range(5):
                print ("----"*3)
                if cfg.base.original_checkpoint == "scratch":
                    net = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=num_classes)
                else:
                    print ("Using pre-trained models from", cfg.base.original_checkpoint)
                    net = smp.Unet(encoder_name="resnet50", encoder_weights=cfg.base.original_checkpoint ,in_channels=1, 
                                   classes=num_classes)


                net.to(device=device)
                _2d_dice, _2d_iou, _3d_iou = eval(cfg = cfg, out_dir = cfg.base.best_valid_model_checkpoint, net = net, device = device, 
                                                  img_scale = (cfg.base.image_shape, cfg.base.image_shape), trial=trial)
                _2d_dices.append(_2d_dice.item())
                _2d_ious.append(_2d_iou.item())
                _3d_ious.append(_3d_iou.item())
            print ("Average performance on best valid set")
            print("2d dice {}, mean {}, std {}".format(_2d_dices, np.mean(_2d_dices), np.std(_2d_dices)))
            print("2d iou {}, mean {}, std {}".format(_2d_ious, np.mean(_2d_ious), np.std(_2d_ious)))
            print("3d iou {}, mean {}, std {}".format(_3d_ious, np.mean(_3d_ious), np.std(_3d_ious)))
                

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
