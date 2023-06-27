from mmdet.apis import set_random_seed
from mmcv import Config

def get_config(base_directory='.'):
  print ("Using base_config_track")
  cfg = Config.fromfile(base_directory + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
  #print(cfg.pretty_text)

  cfg.classes = ("Aortic_enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration", "Lung_Opacity", "Nodule/Mass", "Other_lesion", "Pleural_effusion", "Pleural_thickening", "Pneumothorax", "Pulmonary_fibrosis")

  cfg.data.train.img_prefix = base_directory + '/data/'
  cfg.data.train.ann_file = base_directory + '/data/train_annotations.json'
  cfg.data.train.classes = cfg.classes
  cfg.data.train.type='CocoDatasetSubset'

  img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

  albu_train_transforms = [
    dict(
      type='RandomSizedBBoxSafeCrop',
      height=512,
      width=512,
      erosion_rate=0.2),
  ]

  cfg.data.train.pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
      type='Albu',
      transforms=albu_train_transforms,
      bbox_params=dict(
          type='BboxParams',
          format='pascal_voc',
          label_fields=['gt_labels'],
          min_visibility=0.0,
          filter_lost_elements=True),
      keymap={
          'img': 'image',
          'gt_bboxes': 'bboxes'
      },
      update_pad_shape=False,
      skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
  ]


  cfg.data.train = dict(
    type='ClassBalancedDataset',
    oversample_thr=0.4,
    dataset=cfg.data.train
  )

  cfg.data.val.img_prefix = base_directory + '/data/'
  cfg.data.val.ann_file = base_directory + '/data/valid_annotations.json' 
  cfg.data.val.classes = cfg.classes
  cfg.data.val.type='CocoDataset'

  cfg.data.test.img_prefix = base_directory + '/data/'
  cfg.data.test.ann_file = base_directory + '/data/test_annotations.json'
  cfg.data.test.classes = cfg.classes
  cfg.data.test.type='CocoDataset'

  cfg.model.roi_head.bbox_head.num_classes = 14

  cfg.optimizer.lr = 0.02 / 8
  cfg.lr_config.warmup = None
  cfg.log_config.interval = 10

  # We can set the checkpoint saving interval to reduce the storage cost
  cfg.checkpoint_config.interval = 1

  # Set seed thus the results are more reproducible
  cfg.seed = 1
  set_random_seed(1, deterministic=False)
  cfg.gpu_ids = range(1)

  # we can use here mask_rcnn.
  # cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
  cfg.work_dir = "../trained_weights"


  # One Epoch takes around 18 mins
  cfg.total_epochs = 30
  cfg.runner.max_epochs = 30

  cfg.data.samples_per_gpu = 6

  cfg.log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.

  cfg.workflow = [('train', 1), ('val', 1)]
  cfg.evaluation=dict(classwise=True, metric='bbox')

  return cfg
