# based on https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset#Building-COCO-DATASET

import os
from pathlib import Path
from datetime import datetime
import shutil
from collections import Counter
import warnings
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

warnings.filterwarnings("ignore", category=UserWarning)

def plot_img(img, size=(18, 18), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

def plot_imgs(imgs, cols=2, size=10, is_rgb=True, title="", cmap='gray', img_size=None):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    
def draw_bbox(image, box, label, color, thickness=3):
    alpha = 0.1
    alpha_box = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv.getTextSize(label.upper(), cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                color, -1)
    cv.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv.rectangle(overlay_text, (box[0], box[1]-7-text_height), (box[0]+text_width+2, box[1]),
                (0, 0, 0), -1)
    cv.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                    color, thickness)
    cv.putText(output, label.upper(), (box[0], box[1]-5),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return output

def normalize_bboxes(df):
  df['x_min'] = df.apply(lambda row: (row.x_min)/row.width * 512, axis =1)
  df['y_min'] = df.apply(lambda row: (row.y_min)/row.height * 512, axis =1)

  df['x_max'] = df.apply(lambda row: (row.x_max)/row.width * 512, axis =1)
  df['y_max'] = df.apply(lambda row: (row.y_max)/row.height * 512, axis =1)

  df['x_mid'] = df.apply(lambda row: (row.x_max+row.x_min)/2 * 512, axis =1)
  df['y_mid'] = df.apply(lambda row: (row.y_max+row.y_min)/2 * 512, axis =1)

  df['w'] = df.apply(lambda row: (row.x_max-row.x_min), axis =1)
  df['h'] = df.apply(lambda row: (row.y_max-row.y_min), axis =1)

  df['area'] = df['w']*df['h']
  return df


labels =  [
            "__ignore__",
            "Aortic_enlargement",
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung_Opacity",
            "Nodule/Mass",
            "Other_lesion",
            "Pleural_effusion",
            "Pleural_thickening",
            "Pneumothorax",
            "Pulmonary_fibrosis"
            ]

label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]

viz_labels = labels[1:]

now = datetime.now()

data = dict(
    info=dict(
        description=None,
        url=None,
        version=None,
        year=now.year,
        contributor=None,
        date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
    ),
    licenses=[dict(
        url=None,
        id=0,
        name=None,
    )],
    images=[],
    type='instances',
    annotations=[],
    categories=[],
)

class_name_to_id = {}
for i, each_label in enumerate(labels):
    class_id = i - 1  # starts with -1
    class_name = each_label
    if class_id == -1:
        assert class_name == '__ignore__'
        continue
    class_name_to_id[class_name] = class_id
    data['categories'].append(dict(
        supercategory=None,
        id=class_id,
        name=class_name,
    ))

train_out_dir = 'data/train'
valid_out_dir = 'data/valid'
test_out_dir = 'data/test'

for dir in [train_out_dir, valid_out_dir, test_out_dir]:
  if Path(dir).exists():
    shutil.rmtree(dir)
  os.makedirs(dir)

train_out_file = 'data/train_annotations.json'
valid_out_file = 'data/valid_annotations.json'
test_out_file = 'data/test_annotations.json'

all_images_folder = 'vinbigdata/train'
all_files = os.listdir(all_images_folder)
all_files = np.sort(np.array(all_files))

data_train = data.copy()
data_valid = data.copy()
data_test = data.copy()

for data in [data_train, data_valid, data_test]:
  data['images'] = []
  data['annotations'] = []

all_annotations = pd.read_csv('vinbigdata/train.csv')
all_annotations = all_annotations[all_annotations.class_id != 14]
all_annotations['image_path'] = all_annotations['image_id'].map(lambda id:
  os.path.join(all_images_folder, str(id) + '.png'))
normalize_bboxes(all_annotations)
all_image_paths = all_annotations['image_path'].unique()

np.random.seed(1)

indices = np.arange(len(all_image_paths))
np.random.shuffle(indices)

# train, valid, test
splits = [0.7, 0.1, 0.2]

train_split_index = int(splits[0] * len(indices))
valid_split_index = int((splits[0] + splits[1]) * len(indices))

train_paths = all_image_paths[:train_split_index]
valid_paths = all_image_paths[train_split_index:valid_split_index]
test_paths = all_image_paths[valid_split_index:]

print(f'train: {len(train_paths)}, test: {len(test_paths)}, valid: {len(valid_paths)}')

folders = [train_out_dir, valid_out_dir, test_out_dir]
paths = [train_paths, valid_paths, test_paths]
data_dicts = [data_train, data_valid, data_test]
out_files = [train_out_file, valid_out_file, test_out_file]

# parameters for weighted box fusion
iou_thr = 0.2
skip_box_thr = 0.0001

for (folder, paths, data, out_file) in zip(folders, paths, data_dicts, out_files):
  print(f'Saving to {folder}...')

  viz_images = []

  for i, path in tqdm(enumerate(paths)):
      img_array  = cv.imread(path)
      image_basename = Path(path).stem
      shutil.copy2(path, folder)
      
      ## Add Images to annotation
      data['images'].append(dict(
          license=0,
          url=None,
          file_name=os.path.join(folder.split('/')[-1], image_basename+ '.png'),
          height=img_array.shape[0],
          width=img_array.shape[1],
          date_captured=None,
          id=i
      ))
      
      img_annotations = all_annotations[all_annotations.image_id==image_basename]
      boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
      labels_viz = img_annotations['class_id'].to_numpy().tolist()
      
      ## Visualize Original Bboxes every 500th
      if (i%500==0):
          img_before = img_array.copy()
          for box, label in zip(boxes_viz, labels_viz):
              x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
              color = label2color[int(label)]
              img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
          viz_images.append(img_before)
          
      boxes_list = []
      scores_list = []
      labels_list = []
      weights = []
      
      boxes_single = []
      labels_single = []

      cls_ids = img_annotations['class_id'].unique().tolist()
      
      count_dict = Counter(img_annotations['class_id'].tolist())

      for cid in cls_ids:
          ## Performing Fusing operation only for multiple bboxes with the same label
          if count_dict[cid]==1:
              labels_single.append(cid)
              boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

          else:
              cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
              labels_list.append(cls_list)
              bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
              
              ## Normalizing Bbox by Image Width and Height
              bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
              bbox = np.clip(bbox, 0, 1)
              boxes_list.append(bbox.tolist())
              scores_list.append(np.ones(len(cls_list)).tolist())
              weights.append(1)
      
      ## Perform WBF
      boxes, scores, box_labels = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list,
                                                    labels_list=labels_list, weights=weights,
                                                    iou_thr=iou_thr, skip_box_thr=skip_box_thr)
      
      boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
      boxes = boxes.round(1).tolist()
      box_labels = box_labels.astype(int).tolist()
      boxes.extend(boxes_single)
      box_labels.extend(labels_single)
      
      for box, label in zip(boxes, box_labels):
          x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
          area = round((x_max-x_min)*(y_max-y_min),1)
          bbox =[
                  round(x_min, 1),
                  round(y_min, 1),
                  round((x_max-x_min), 1),
                  round((y_max-y_min), 1)
                  ]
          
          data['annotations'].append(dict( id=len(data['annotations']), image_id=i,
                                              category_id=int(label), area=area, bbox=bbox,
                                              iscrowd=0))
          
      ## Visualize Bboxes after operation every 500th
      if (i%500==0):
          img_after = img_array.copy()
          for box, label in zip(boxes, box_labels):
              color = label2color[int(label)]
              img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
          viz_images.append(img_after)

  plot_imgs(viz_images, cmap=None, size=40)
  plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=15)
  plt.figtext(0.73, 0.9,"WBF", va="top", ha="center", size=15)
  plt.show()
                
  with open(out_file, 'w') as f:
      json.dump(data, f, indent=4)
