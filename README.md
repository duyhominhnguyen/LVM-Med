# LVM-Med
Release LMV-Med pre-trained models. We demonstrate downstream tasks on 2D-3D segmentations, linear/fully finetuning image classification, and object detection.  

## 1. Pretrained models
<table>
  <tr>
    <th>Arch</th>
    <th>Params (M)</th>
    <th> 2D Segmentation (Dice) </th>
    <th> 3D Segmentation (3D IoU) </th>
    <th>Weights</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>25.5M</td>
    <td>83.05</td>
    <td>79.02</td>
    <td> <a href="https://drive.google.com/file/d/11Uamq4bT_AbTf8sigIctIAnQJN4EethW/view?usp=sharing">backbone</a> </td>
  </tr>
  <tr>
    <td>ViT-B</td>
    <td>86.0M</td>
    <td>85.80</td>
    <td>73.85</td>
    <td> <a href="https://drive.google.com/file/d/14bX8wdw-c3VUw3XPAtFMB-wFE03q0eCi/view?usp=sharing">backbone</a> </td>
  </tr>
</table>

After installing the pre-trained models, please place them in [`checkpoints`](/checkpoints/) folder to use them. 

- For **Resnet-50**, we run **end-to-end** segmentation
- For **ViT-B**, we run **prompt-based** segmentation
- The code and pre-trained model for **ViT-B** **end-to-end** segmentation will be uploaded soon!!!

## 2. Project setup

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

To set up our project, run the following command:

```bash
git clone https://github.com/duyhominhnguyen/LVM-Med
cd LVM-Med
conda env create -f lvm_med.yml
conda activate lvm_med
```

For running **Fine-tune for downstream tasks using ResNet-50** in **Section 4f**, we utilize Unet from `segmentation-models-pytorch` package. To install this library we do the following tasks: 

1. `git clone https://github.com/qubvel/segmentation_models.pytorch.git` and `cd segmentation_models_pytorch`
2. Access this path in the smp package: `/encoders/resnet.py` and add your download pre-trained weight (you can see our example in [`segmentation_models_pytorch/encoders
/resnet.py`](segmentation_models_pytorch/encoders/resnet.py) )
3. Then `pip install segmentation-models-pytorch` to install its dependencies



## 3. Prepare dataset
First you should download the respective dataset that you need to run to the [`dataset_demo`](/dataset_demo/) folder. To get as close results as your work as possible, you could prepare some of our specific dataset (which are not pre-distributed) the same way as we do:
```bash
python prepare_dataset.py -ds [dataset_name]
```
such that: `dataset_name` is the name of dataset that you would like to prepare. After that, you should change paths to your loaded dataset on our pre-defined yaml file in [`dataloader/yaml_data`](/dataloader/yaml_data/).

Currently support for `Kvasir`, `BUID`, `FGADR`, `MMWHS_MR_Heart` and `MMWHS_CT_Heart`

## 4. How to reproduce our results ?

### a. End-to-end classification for downstream tasks
```bash
# Fully fine-tuned with 1 FCN
python train_classification.py -c ./dataloader/yaml_data/fgadr_endtoend_R50_non_frozen_1_fcn.yml

# Fully fine-tuned with multiple FCNs
python train_classification.py -c ./dataloader/yaml_data/fgadr_endtoend_R50_non_frozen_fcns.yml

# Freeze all and fine-tune 1-layer FCN only 
python train_classification.py -c ./dataloader/yaml_data/fgadr_endtoend_R50_frozen_1_fcn.yml

# Freeze all and fine-tune multi-layer FCN only
python train_classification.py -c ./dataloader/yaml_data/fgadr_endtoend_R50_frozen_fcns.yml
```
### b. End-to-end object detection for downstream tasks
You can refer to [`Object Detection`](/Object_Detection) folder for more details.

### c. Zero-shot prompt-based segmentation with Segment Anything Model (SAM) for downstream tasks
```bash
python3 zero_shot_segmentation.py -c dataloader/yaml_data/buid_sam.yml
```

### d. Prompt-based segmentation with Fine-tune SAM (MedSAM) for downstream tasks
You could also see the examples of [`Prompt_Demo.ipynb`](/notebook/Prompt_Demo.ipynb) for results visualization using prompt-based MedSAM.
#### Train
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml 
```
#### Inference
```bash
python3 medsam.py -c dataloader/yaml_data/buid_sam.yml -test
```

### e. Prompt-based segmentation with LVM-Med (encoder) + MedSAM for downstream tasks
The difference in yaml file between LVM-Med + MedSAM and the original MedSAM is just the way you save the trained model. Hence you could either use the MedSAM's yaml file or create a new one, it would make no difference in the final performance.     

For results visualization with LVM-Med as encoder, you could check our examples at [`LVMMed_Encoder_Prompt_Demo.ipynb`](/notebook/LVMMed_Encoder_Prompt_Demo.ipynb)

```bash
python3 medsam.py -c dataloader/yaml_data/buid_lvm_med_sam.yml -lvm_encoder workdir/pretrained/vit_b_largescale_dim256.pth
```

### f. LVM-Med 
#### Fine-tune for downstream tasks using ResNet-50

```bash
python train_segmentation.py -c ./dataloader/yaml_data/buid_endtoend_R50.yml
```
#### Fine-tune for downstream tasks using SAM's VIT
```bash
python train_segmentation.py -c ./dataloader/yaml_data/buid_endtoend_SAM_VIT.yml
```
