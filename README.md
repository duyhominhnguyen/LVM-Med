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
  <tr>
    <th>Resnet50</th>
    <th> 90% (%) </th>
    <th> 90% (Dice) </th>
    <th> [GoogleDrive](https://drive.google.com/file/d/1I08oJlXNtDe8jJPxHkroxUi7lYX2lhVc/view?usp=sharing) </th>
  </tr>
</table>

## 2. Project setup
```bash
git clone https://github.com/duyhominhnguyen/LVM-Med
cd LVM-Med
conda env create -f lvm_med.yml
conda activate lvm_med
```

## 3. Prepare dataset
First you should download the respective dataset that you need to run to the [`dataset_demo`](/dataset_demo/) folder. To get as close results as your work as possible, you could prepare some of our specific dataset (which are not pre-distributed) the same way as we do:
```bash
python prepare_dataset.py -ds [dataset_name]
```
such that: `dataset_name` is the name of dataset that you would like to prepare. After that, you should change paths to your loaded dataset on our pre-defined yaml file in [`dataloader/yaml_data`](/dataloader/yaml_data/).

Currently support for `Kvasir`, `BUID`, `FGADR`, `MMWHS_MR_Heart` and `MMWHS_CT_Heart`

## 4. How to reproduce our results ?

### Zero-shot prompt-based segmentation with Segment Anything Model (SAM) for downstream tasks
```bash
python3 zero_shot_segmentation.py -c dataloader/yaml_data/buid_sam.yml
```

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
The difference in yaml file between LVM-Med + MedSAM and the original MedSAM is just the way you save the trained model. Hence you could either use the MedSAM's yaml file or create a new one, it would make no difference in the final performance.     

For results visualization with LVM-Med as encoder, you could check our examples at [`LVMMed_Encoder_Prompt_Demo.ipynb`](/notebook/LVMMed_Encoder_Prompt_Demo.ipynb)

```bash
python3 medsam.py -c dataloader/yaml_data/buid_lvm_med_sam.yml -lvm_encoder workdir/pretrained/vit_b_largescale_dim256.pth
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
