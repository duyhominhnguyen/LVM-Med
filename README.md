# LVM-Med
Release LMV-Med pre-trained models. We demonstrate downstream tasks on 2D-3D segmentations, linear/fully finetuning image classification, and object detection.  

## 1. Pretrained models
<table>
  <tr>
    <th>Arch</th>
    <th> Classification (%) </th>
    <th> 2D Segmentation (Dice) </th>
    <th colspan="6">Weights</th>
  </tr>
</table>

## 2. How to reproduce our results ?
### Prompt-based segmentation with Fine-tune SAM (MedSAM) for downstream tasks
You could also see the examples of [`Prompt_Demo.ipynb`](/notebook/Prompt_Demo.ipynb) for results visualization using prompt-based MedSAM.
#### Train
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml 
```
#### Inference
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml -test
```

### Prompt-based segmentation with LVM-Med (encoder) + MedSAM for downstream tasks
```bash

```

### LVM-Med 
#### Fine-tune for downstream tasks using ResNet-50
```bash
python train_segmentation.py -c ./dataloader/yaml_data/buid_endtoend_R50.yml
```
#### Fine-tune for downstream tasks using SAM's VIT
```bash
python train_segmentation.py -c ./dataloader/yaml_data/buid_endtoend_SAM_VIT.yml
```
