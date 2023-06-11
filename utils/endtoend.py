import torch
from torch import Tensor
import numpy as np
import glob
import pandas as pd

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    if input.dim() == 3:
        return dice_coeff(input, target, reduce_batch_first, epsilon)
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def iou_2d(outputs: torch.Tensor, labels: torch.Tensor, reduce_batch_first: bool =False, epsilon=1e-6):
    if outputs.dim() == 2 or reduce_batch_first:
        inter = torch.dot(outputs.reshape(-1), labels.reshape(-1))
        union = outputs.sum() + labels.sum() - inter
        return (inter + epsilon)/ (union + epsilon)
    else:
        iou = 0 
        for idx in range(outputs.size(0)):
            iou += iou_2d(outputs[idx], labels[idx])
        return iou/outputs.size(0)

def multiclass_iou(outputs: torch.Tensor, labels: torch.Tensor, reduce_batch_first: bool =False):
    assert outputs.size() == labels.size()
    if outputs.dim() == 3:
        return iou_2d(outputs, labels, reduce_batch_first)
    iou = 0
    for cidx in range(outputs.size(1)):
        iou += iou_2d(outputs[:,cidx,...], labels[:, cidx, ...], reduce_batch_first)
    return iou/outputs.size(1)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


