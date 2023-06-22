## Training Faster RCNN model using LVM-Med-R50

# activate conda environment
conda activate vin

# convert dataset to Coco format
cd /home/caduser/KOTORI/vin-ssl/LVM-Med-classification
python convert_to_coco.py # -> note, please check again if this file is correct. Don't run this file again.

# set train, valid, test folders
cd /home/caduser/KOTORI/vin-ssl/LVM-Med-classification
edit file base_config_track.py at lines
+ 11, 12 for training set
+ 60, 61 for valid set
+ 65, 66 for test set
+ 86 for folder store models.

# train model and test
cd /home/caduser/KOTORI/vin-ssl/LVM-Med-classification
bash command.sh

# train again from current epochs, run
CUDA_VISIBLE_DEVICES=5 python finetune_object_detect.py --experiment-name 'lvm-med-r50' --weight-path /home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/graph_match/converted_vissl_ssl_graph_large.torch --batch-size 16 --optim adam --clip 1 --lr 0.0001 --epochs 40 --labeled-dataset-percent 1.0 --resume
