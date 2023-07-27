import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.endtoend import multiclass_dice_coeff, multiclass_iou
num_classes = 8

def evaluate(net, dataloader, device, eval_class):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask_ete']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        #mask_true[mask_true == 4] = 3
        #mask_true[mask_true > 4] = 0
        mask_true_vector = F.one_hot(mask_true, num_classes).permute(0, 3 , 1, 2).float()
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)
            mask_pred_vector = F.one_hot(mask_pred, num_classes).permute(0, 3 , 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred_vector[:, eval_class, ...], mask_true_vector[:, eval_class, ...],
                                                    reduce_batch_first=False)
            iou_score += multiclass_iou(mask_pred_vector[:,eval_class, ...], mask_true_vector[:, eval_class, ...])

    net.train()
    return dice_score / num_val_batches, iou_score/ num_val_batches


def evaluate_3d_iou(net, dataset, device, eval_class):
    net.eval()
    iou_score = 0
    # iterate over the validation set
    num_items = 0
    for image_3d in tqdm(dataset.get_3d_iter(), desc='3D Evaluation', unit='image(s)', leave=False):
        image, mask_true = image_3d['image'], image_3d['mask_ete']
        num_items += 1
        # move images and labels to correct device and type

        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_vector = F.one_hot(mask_true, num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)
            mask_pred_vector = F.one_hot(mask_pred, num_classes).permute(0, 3, 1, 2).float()
            iou_score += multiclass_iou(mask_pred_vector[:, eval_class, ...], mask_true_vector[:, eval_class, ...], reduce_batch_first=True)
    net.train()
    return iou_score/num_items

def evaluate_3d_iou_large(net, dataset, device, eval_class):
    net.eval()
    iou_score = 0
    # iterate over the validation set
    num_items = 0
    for image_3d in tqdm(dataset.get_3d_iter(), desc='3D Evaluation', unit='image(s)', leave=False):
        image, mask_true = image_3d['image'], image_3d['mask']
        num_items += 1
        # move images and labels to correct device and type

        image = image.to(device=device)
        mask_true = mask_true.to(device=device)
        mask_true_vector = F.one_hot(mask_true, num_classes).permute(0, 3, 1, 2).float()

        net.to(device=device)
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)
            mask_pred_vector = F.one_hot(mask_pred, num_classes).permute(0, 3, 1, 2).float()
            iou_score += multiclass_iou(mask_pred_vector[:, eval_class, ...], mask_true_vector[:, eval_class, ...], reduce_batch_first=True)
    net.train()
    return iou_score/num_items

def evaluate_3d_iou_fast(net, dataset, device, eval_class):
    """
    This function is similar as evaluate_3d_iou but get a batch size in shape [batch_size, dimension, W, H]
    :param net: 
    :param dataset: 
    :param device: 
    :param eval_class: 
    :return: 
    """
    net.eval()
    iou_score = 0
    # iterate over the validation set
    num_items = 0
    for image_3d in tqdm(dataset, desc='3D Evaluation', unit='image(s)', leave=False):
        image, mask_true = image_3d['image'][0], image_3d['mask'][0]
        # print ("Image and mask shapes in 3D evaluation are {}, {}".format(image.shape, mask_true.shape))
        num_items += 1
        # move images and labels to correct device and type

        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_vector = F.one_hot(mask_true, num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)
            mask_pred_vector = F.one_hot(mask_pred, num_classes).permute(0, 3, 1, 2).float()
            iou_score += multiclass_iou(mask_pred_vector[:, eval_class, ...], mask_true_vector[:, eval_class, ...], reduce_batch_first=True)
    net.train()
    return iou_score/num_items

def evaluate_3d_dice(net, dataset, device, eval_class):
    net.eval()
    dice_score = 0
    # iterate over the validation set
    num_items = 0
    for image_3d in tqdm(dataset.get_3d_iter(), desc='3D Evaluation', unit='image(s)', leave=False):
        image, mask_true = image_3d['image'], image_3d['mask_ete']
        num_items += 1
        # move images and labels to correct device and type

        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_vector = F.one_hot(mask_true, num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = mask_pred.argmax(dim=1)
            mask_pred_vector = F.one_hot(mask_pred, num_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(mask_pred_vector[:, eval_class, ...], mask_true_vector[:, eval_class, ...], reduce_batch_first=True)
    net.train()
    return dice_score/num_items
