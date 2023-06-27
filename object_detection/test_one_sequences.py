import sys
import os
from natsort import natsorted
import argparse
from collections import OrderedDict
import io
import contextlib
import itertools
import numpy as np
import mmcv
from mmdet.datasets.api_wrappers import COCO, COCOeval
sys.path.append('/home/caduser/KOTORI/vin-ssl/source')
os.chdir('/home/caduser/KOTORI/vin-ssl/source')

from base_config_track import get_config
from mmdet_tools import mmdet_test

def print_log(msg, logger):
    pass
    #print(msg)

def evaluate(dataset, results, metric='bbox', logger=None, jsonfile_prefix=None, classwise=False, proposal_nums=(100, 300, 1000), iou_thrs=None, metric_items=None):
    """Evaluation in COCO protocol.
    Args:
        results (list[list | tuple]): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated. Options are
            'bbox', 'segm', 'proposal', 'proposal_fast'.
        logger (logging.Logger | str | None): Logger used for printing
            related information during evaluation. Default: None.
        jsonfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        classwise (bool): Whether to evaluating the AP for each class.
        proposal_nums (Sequence[int]): Proposal number used for evaluating
            recalls, such as recall@100, recall@1000.
            Default: (100, 300, 1000).
        iou_thrs (Sequence[float], optional): IoU threshold used for
            evaluating recalls/mAPs. If set to a list, the average of all
            IoUs will also be computed. If not specified, [0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
            Default: None.
        metric_items (list[str] | str, optional): Metric items that will
            be returned. If not specified, ``['AR@100', 'AR@300',
            'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
            used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
            'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
            ``metric=='bbox' or metric=='segm'``.
    Returns:
        dict[str, float]: COCO style evaluation metric.
    """
    metrics = metric if isinstance(metric, list) else [metric]
    allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
    if iou_thrs is None:
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    if metric_items is not None:
        if not isinstance(metric_items, list):
            metric_items = [metric_items]

    result_files, tmp_dir = dataset.format_results(results, jsonfile_prefix)

    eval_results = OrderedDict()
    cocoGt = dataset.coco

    results_per_category = []

    for metric in metrics:
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        if metric == 'proposal_fast':
            ar = dataset.fast_eval_recall(
                results, proposal_nums, iou_thrs, logger='silent')
            log_msg = []
            for i, num in enumerate(proposal_nums):
                eval_results[f'AR@{num}'] = ar[i]
                log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            continue

        iou_type = 'bbox' if metric == 'proposal' else metric
        if metric not in result_files:
            raise KeyError(f'{metric} is not in results')
        try:
            predictions = mmcv.load(result_files[metric])
            if iou_type == 'segm':
                # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                # When evaluating mask AP, if the results contain bbox,
                # cocoapi will use the box area instead of the mask area
                # for calculating the instance area. Though the overall AP
                # is not affected, this leads to different
                # small/medium/large mask AP results.
                for x in predictions:
                    x.pop('bbox')
                warnings.simplefilter('once')
                warnings.warn(
                    'The key "bbox" is deleted for more accurate mask AP '
                    'of small/medium/large instances since v2.12.0. This '
                    'does not change the overall mAP calculation.',
                    UserWarning)
            cocoDt = cocoGt.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger,
                level=logging.ERROR)
            break

        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.catIds = dataset.cat_ids
        cocoEval.params.imgIds = dataset.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')

        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                    'AR_m@1000', 'AR_l@1000'
                ]

            for item in metric_items:
                val = float(
                    f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)
            
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(dataset.cat_ids) == precisions.shape[2]

                for idx, catId in enumerate(dataset.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = dataset.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', float(ap)))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return eval_results, results_per_category

# define parse
def get_args():
    parser = argparse.ArgumentParser(description='Test trained object detection model')
    parser.add_argument(
        '--experiment_name', '-exp-name', type=str, default='no-exp',help='providing folder store checkpoint models')
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    experiment_name = args.experiment_name
    print ("**********" * 3)
    print ('Staring evaluation process')
    checkpoints = os.listdir(os.path.join('../trained_weights', experiment_name))
    checkpoints = natsorted(checkpoints)
    checkpoints = [p for p in checkpoints if 'epoch_' in p]
    # checkpoint = os.path.join('../trained_weights', experiment_name, checkpoints[-1])

    selected_checkpoints = checkpoints[-1:] # change the number of models want to infer here.
    dict_results = {}
    valid_dict_results = {}
    eval_on_valid = False

    for checkpoint_name in selected_checkpoints:
        print ('-----'*5)
        print ('Processing for checkpoint', checkpoint_name)
        checkpoint = os.path.join('../trained_weights', experiment_name, checkpoint_name)

        results = {}
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        results_avg = []
        results_avg_ar = []
        results_classwise = []

        cfg = get_config()

        if eval_on_valid:
            cfg.data.test['img_prefix'] = './data/'  # uncomment lines 267-268 for inference on validation set
            cfg.data.test['ann_file'] = './data/valid_annotations.json'

        args_result = argparse.Namespace(eval='bbox', out='results/' + experiment_name + '.pkl', checkpoint=None, work_dir=results_dir, fuse_conv_bn=None,
               gpu_ids=None, format_only=None, show=None, show_dir=None, show_score_thr=0.3, gpu_collect=None,
               tmpdir=None, cfg_options=None, options=None, launcher='none', eval_options=None, local_rank=0)

        dataset, outputs = mmdet_test.get_outputs(cfg, checkpoint, args_result)

        metrics, results_per_category = evaluate(dataset, outputs, metric='bbox', classwise=True) #, iou_thrs=[0.5])
        metrics_ar, _ = evaluate(dataset, outputs, metric='proposal')
        results_avg.append([experiment_name, metrics])
        results_avg_ar.append([experiment_name, metrics_ar])
        results_classwise.append([experiment_name, OrderedDict(results_per_category)])

        print('--------------------------------')
        valid_dict_results[checkpoint_name] = []
        print('Average Precision')
        print(list(results_avg[0][1].keys())[:-1])

        valid_dict_results[checkpoint_name].append(list(results_avg[0][1].keys())[:-1])  # append output to valid_dict_results

        for res in results_avg:
            print([res[0], list(res[1].values())[:-1]])
            valid_dict_results[checkpoint_name].append([res[0], list(res[1].values())[:-1]])

        dict_results[checkpoint_name] = list(results_avg[0][1].values())[1]
    print ("Results on testing set")
    print (valid_dict_results)
    print("**********" * 3)
