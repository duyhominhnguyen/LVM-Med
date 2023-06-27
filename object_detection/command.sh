## Training Faster RCNN model using LVM-Med-R50
CUDA_VISIBLE_DEVICES=5 python finetune_with_path_modify_test_eval.py --experiment-name 'lvm-med-r50' --weight-path ../lvm_med_weights/lvmmed_resnet.torch --batch-size 16 --optim adam --clip 1 --lr 0.0001 --epochs 40 --labeled-dataset-percent 1.0
# test model
python test_one_sequences.py -exp-name 'lvm-med-r50'
