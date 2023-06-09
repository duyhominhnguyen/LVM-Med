# LVM-Med
Release LMV-Med pre-trained models. We demonstrate downstream tasks on 2D-3D segmentations, linear/fully finetuning image classification, and object detection.  

## How to reproduce our results ?
### Fine-tune SAM (MedSAM) for downstream tasks
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml 
```
### Inference on downstream tasks with SAM 
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml -test
```
