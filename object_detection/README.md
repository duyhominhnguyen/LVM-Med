# Training Faster RCNN model using LVM-Med (R50)

## 1. Activate conda environment
```bash
conda activate lvm_med 
```

## 2. Convert dataset to Coco format
We illustrate LVM-Med ResNet-50 for VinDr dataset, which detects 14 different regions in X-ray images.
You can download the dataset from this link [`VinDr`](https://www.kaggle.com/datasets/awsaf49/vinbigdata-512-image-dataset) and put the folder vinbigdata into the folder object_detection. To build the dataset, after downloading the dataset, you can refer to the script ```convert_to_coco.py``` inside the folder object_detection and run it.
```bash
python convert_to_coco.py # Note, please check links inside the code in lines 146 and 158 to build the dataset correctly
```

## 3. Set train, valid, test folders
Edit [`base_config_track.py`](/Object_Detection/base_config_track.py) at:
+ Lines `11`, `12` for training set
+ Lines `60`, `61` for valid set
+ Lines `65`, `66` for test set
+ Lines `86` for folder store models.

## 4. Train model and test
```bash
bash command.sh
```

## 5. Train from current epochs:
```bash
CUDA_VISIBLE_DEVICES=5 python finetune_with_path_modify_test_eval.py --experiment-name 'lvm-med-r50' --weight-path ../lvm_med_weights/lvmmed_resnet.torch --batch-size 16 --optim adam --clip 1 --lr 0.0001 --epochs 40 --labeled-dataset-percent 1.0 --resume
```

