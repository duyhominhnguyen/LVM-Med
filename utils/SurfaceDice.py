import torch
import numpy as np
import scipy.ndimage 

def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`. 
    
    Args:
        mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
        the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    
    if torch.all(mask_gt == 0): # If the present mask is empty (slice contains no information)
        if np.all(mask_pred == 0): # If model segments nothing
           return torch.tensor(1) # Then dice score is 1
        return torch.tensor(0) # Else dice score is 0
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum
    
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
    for cidx in range(outputs.shape[0]):
        iou += iou_2d(outputs[cidx], labels[cidx], reduce_batch_first)
    return iou/outputs.size(0)