# LVM-Med
Release LMV-Med pre-trained models. We demonstrate downstream tasks on 2D-3D segmentations, linear/fully finetuning image classification, and object detection.  

## How to reproduce our results ?
### Fine-tune SAM (MedSAM) for 2D downstream tasks
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml 
```
### Inference on 2D downstream tasks with SAM 
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml -test
```
