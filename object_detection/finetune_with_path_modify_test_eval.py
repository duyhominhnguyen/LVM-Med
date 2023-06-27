import argparse
import sys
from datetime import datetime
import os
# os.chdir('/home/caduser/KOTORI/vin-ssl/source')
# sys.path.append('/home/caduser/KOTORI/vin-ssl/source')
import copy
import shutil

from natsort import natsorted
from collections import OrderedDict
import torch

from mmdet.datasets import build_dataset, CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.builder import DATASETS
from mmdet.models import build_detector
from mmdet.apis import train_detector
from base_config_track import get_config

@DATASETS.register_module()
class CocoDatasetSubset(CocoDataset):
  """
  A subclass of MMDetection's default COCO dataset which has the ability
  to take the first or last n% of the original dataset. Set either
  take_first_percent or take_last_percent to a value greater than 0.
  """
  def __init__(self, *args, take_first_percent=-1, take_last_percent=-1, **kwargs):
    self.take_first_percent = take_first_percent
    self.take_last_percent = take_last_percent
    super().__init__(*args, **kwargs)

  def load_annotations(self, ann_file):
    """Load annotation from COCO style annotation file.

    Args:
        ann_file (str): Path of annotation file.

    Returns:
        list[dict]: Annotation info from COCO api.
    """
    assert self.take_first_percent > 0 or self.take_last_percent > 0, f'take_first_percent: {self.take_first_percent}, take_last_percent: {self.take_first_percent}'
    assert(self.take_first_percent > 0 if self.take_last_percent <= 0 else self.take_first_percent <= 0)

    self.coco = COCO(ann_file)
    # The order of returned `cat_ids` will not
    # change with the order of the CLASSES
    self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

    self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
    self.img_ids = self.coco.get_img_ids()

    original_count = len(self.img_ids)

    # make a subset
    if self.take_first_percent > 0:
      first_n = True
      count = int(len(self.img_ids) * self.take_first_percent)
      self.img_ids = self.img_ids[:count]
    elif self.take_last_percent > 0:
      first_n = False
      count = int(len(self.img_ids) * self.take_last_percent)
      self.img_ids = self.img_ids[-count:]

    new_count = len(self.img_ids)

    print(f'Taking {"first" if first_n else "last"} {new_count} of original dataset ({original_count}), ({(new_count / original_count) * 100})%')

    data_infos = []
    total_ann_ids = []
    for i in self.img_ids:
        info = self.coco.load_imgs([i])[0]
        info['filename'] = info['file_name']
        data_infos.append(info)
        ann_ids = self.coco.get_ann_ids(img_ids=[i])
        total_ann_ids.extend(ann_ids)
    assert len(set(total_ann_ids)) == len(
        total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
    return data_infos


def get_training_datasets(labeled_dataset_percent, base_directory = '.'):
  cfg = get_config(base_directory)
  cfg.data.train['dataset']['take_last_percent'] = labeled_dataset_percent
  dataset_finetune = build_dataset(cfg.data.train)

  if labeled_dataset_percent < 1:
    cfg.data.train['dataset']['take_last_percent'] = -1
    cfg.data.train['dataset']['take_first_percent'] = 1 - labeled_dataset_percent
    dataset_pretrain = build_dataset(cfg.data.train)
  else:
    dataset_pretrain = None

  return dataset_pretrain, dataset_finetune

def train(experiment_name, weight_path, labeled_dataset_percent, epochs, batch_size, optim, clip, lr, resume):
  cfg = get_config()
  cfg.total_epochs = epochs
  cfg.runner.max_epochs = epochs
  cfg.data.samples_per_gpu = batch_size

  if optim=='adam':
    cfg.optimizer = dict(type='Adam', lr=lr, weight_decay=0.0001)
  else:
    cfg.optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001)

  if clip:
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=clip, norm_type=2))
  else:
    cfg.optimizer_config = dict(grad_clip=None)

  cfg.work_dir += '/' + experiment_name

  logs_folder = os.path.join(cfg.work_dir, 'tf_logs')

  if resume:
    checkpoints = os.listdir(cfg.work_dir)
    checkpoints = natsorted(checkpoints)
    checkpoints = [p for p in checkpoints if 'epoch_' in p]
    checkpoint = os.path.join(cfg.work_dir, checkpoints[-1])
    cfg.resume_from = checkpoint
    print ('initialize learning rate again')
    cfg.optimizer.lr = lr
    print (cfg.optimizer)
  else:
    if (os.path.exists(logs_folder)):
      shutil.rmtree(logs_folder)

    print(cfg.model.backbone.init_cfg)

    if (os.path.exists(weight_path)):
      state_dict = torch.load(weight_path)
      new_state_dict = OrderedDict()
      for k, v in state_dict.items():
        name = 'backbone.' + k
        new_state_dict[name] = v
      torch.save(new_state_dict, 'tmp.pth')
      cfg.load_from = 'tmp.pth'
      print('Loading pretrained backbone from ' + weight_path)

  _, train_dataset = get_training_datasets(labeled_dataset_percent)
  
  model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'))
  datasets = [train_dataset]
  cfg.workflow = [('train', 1)]
  cfg.device = 'cuda'
  # train model
  train_detector(model, datasets, cfg, distributed=False, validate=True)

def parse_args():
  parser = argparse.ArgumentParser(description='Train using MMDet and Lightly SSL')
  parser.add_argument('--experiment-name', default='no-exp')
  parser.add_argument('--weight-path', type=str, required=True)
  parser.add_argument('--labeled-dataset-percent', type=float, default=1)
  parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train',
  )
  parser.add_argument(
    '--batch-size',
    type=int,
    default=6,
  )
  parser.add_argument(
    '--optim',
    type=str,
    default='sgd',
  )
  parser.add_argument(
    '--clip',
    type=float,
    default=0,
  )
  parser.add_argument(
    '--lr',
    type=float,
    default=0.02 / 8,
  )
  parser.add_argument(
    '--resume',
    default=False,
    action='store_true',
    help='resume training from last checkpoint in work dir'
  )
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  train(**vars(args))

if __name__ == '__main__':
  main()
  os.remove('tmp.pth')
