## Training Faster RCNN model using LVM-Med-R50
CUDA_VISIBLE_DEVICES=5 python finetune_with_path_modify_test_eval.py --experiment-name 'lvm-med-r50' --weight-path /home/caduser/KOTORI/WEIGHTS/VISSL_pretrained/graph_match/converted_vissl_ssl_graph_large.torch --batch-size 16 --optim adam --clip 1 --lr 0.0001 --epochs 40 --labeled-dataset-percent 1.0
# test model
python test_one_sequences.py -exp-name 'lvm-med-r50'
