import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)

        #Class wise recall 
        metric[f'recall_roi_{cur_thresh}_car'] += ret_dict.get(f'roi_{cur_thresh}_car', 0)
        metric[f'recall_roi_{cur_thresh}_ped'] += ret_dict.get(f'roi_{cur_thresh}_ped', 0)
        metric[f'recall_roi_{cur_thresh}_cyc'] += ret_dict.get(f'roi_{cur_thresh}_cyc', 0)
        
        metric[f'recall_rcnn_{cur_thresh}_car'] += ret_dict.get(f'rcnn_{cur_thresh}_car', 0)
        metric[f'recall_rcnn_{cur_thresh}_ped'] += ret_dict.get(f'rcnn_{cur_thresh}_ped', 0)
        metric[f'recall_rcnn_{cur_thresh}_cyc'] += ret_dict.get(f'rcnn_{cur_thresh}_cyc', 0)


    metric['gt_num'] += ret_dict.get('gt', 0)
    metric['gt_car_num'] += ret_dict.get('num_car_gt', 0)
    metric['gt_ped_num'] += ret_dict.get('num_ped_gt', 0)
    metric['gt_cyc_num'] += ret_dict.get('num_cyc_gt', 0)

    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # metric contains:
    # gt_num: total num all relevant gt boxes in the whole dataset 
    # recall_roi_0.3: num of all gt boxes which overlapped with the roi (1st stage) predicted boxes with iou3d > 0.3
    # recall_rcnn_0.3: num of all gt boxes which overlapped with the rcnn (final stage) predicted boxes with iou3d > 0.3
    # recall_roi_0.5:
    # recall_rnn_0.5:
    # recall_roi_0.7: (hardest so fewest boxes)
    # recall_rnn_0.7: (hardest so fewest boxes)
    metric = {
        'gt_num': 0,
        'gt_car_num': 0,
        'gt_ped_num': 0,
        'gt_cyc_num': 0
    }
    classes = ['car', 'ped', 'cyc']
    train_loss_dict = {}
    test_loss_dict = {'before_nms': {}, 'after_nms': {}}
    score_threshs = [0.1, 0.5, 0.8]
    conf_matrix = {}

    for cls in classes:
        for score in score_threshs:
            conf_matrix[f'{cls}_{score}'] = {'tp': 0, 'fp_due_to_wrong_cls': 0, 'fp_due_to_no_overlap_but_high_confidence': 0,
            'fn_due_to_low_score_or_wrong_cls': 0, 
            'fn_due_to_no_overlap': 0, 
            'fn_due_to_low_score_and_wrong_cls': 0,
            'fn_due_to_low_score': 0,
            'fn_due_to_wrong_cls': 0}

    for cls in classes:
        # #Train loss
        # train_loss_dict[f'point_loss_cls_{cls}'] = 0
        # train_loss_dict[f'point_loss_box_{cls}'] = 0
        # train_loss_dict[f'point_loss_num_{cls}_pts'] = 0
        
        
        # train_loss_dict[f'rcnn_loss_reg_{cls}'] = 0
        # train_loss_dict[f'rcnn_loss_corner_{cls}'] = 0
        # train_loss_dict[f'rcnn_loss_num_{cls}_boxes'] = 0

        # train_loss_dict[f'rcnn_loss_cls_{cls}'] = 0
        # train_loss_dict[f'rcnn_loss_num_{cls}_rois'] = 0

        for key in ['before_nms', 'after_nms']: #, 
            #Test loss
            test_loss_dict[key][f'gt_wise_cls_loss_{cls}'] = []
            test_loss_dict[key][f'gt_wise_objectness_loss_{cls}'] = []
            test_loss_dict[key][f'gt_wise_box_position_loss_{cls}'] = []
            test_loss_dict[key][f'gt_wise_box_dim_loss_{cls}'] = []

            # test_loss_dict[key][f'pred_wise_cls_loss_{cls}'] = []
            # test_loss_dict[key][f'pred_wise_objectness_loss_{cls}'] = []
            # test_loss_dict[key][f'pred_wise_box_position_loss_{cls}'] = []
            # test_loss_dict[key][f'pred_wise_box_dim_loss_{cls}'] = []


    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        #Class wise recall
        for cls in classes:
            metric[f'recall_roi_{cur_thresh}_{cls}'] = 0
            metric[f'recall_rcnn_{cur_thresh}_{cls}'] = 0


    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = [] # len: number of frames in dataset. Contains pred dicts for the whole dataset (see generate_single_sample_dict: how one pc's pred  dect is stored)

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        
        # Forward pass
        with torch.no_grad():
            pred_dicts, ret_dict, train_l_dict, test_l_dict, conf_matrix_dict = model(batch_dict) #batch_dict['points']:(16384x2, 5) [b_id, xyzi] #batch_dict['gt_boxes']: (2, num boxes, 8) [x, y, z, dx dy dz, r, class_id] class_id can be 1,2,3
        disp_dict = {}

        # Add ret_dict i.e. recall_dict for this batch to metric dict 
        statistics_info(cfg, ret_dict, metric, disp_dict)
        # Convert 3d boxes in lidar frame to 3d boxes in camera frame and store as kitti annos format
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos

        # Add train losses
        if train_l_dict is not None:
            for key, val in train_l_dict.items():
                train_loss_dict[key] += val

        # Add test losses
        for key_i in ['before_nms', 'after_nms']: #, 
            for key_j, val in test_l_dict[key_i].items():
                if val > 0:
                    test_loss_dict[key_i][key_j] += [val]
        
        # Add confusion matrix
        for cls_score_key, cls_score_dict in conf_matrix_dict.items():
            for key, val in cls_score_dict.items():
                conf_matrix[cls_score_key][key] += val

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']

    #Print conf matrix:
    logger.info('*************** Confusion Matrix *****************')
    for cls_score_key, cls_score_dict in conf_matrix.items():
        for key, val in cls_score_dict.items():
            logger.info(f'{cls_score_key}_{key}: {val}')
            
        prec = cls_score_dict['tp'] / (cls_score_dict['tp'] + cls_score_dict['fp_due_to_wrong_cls'] + cls_score_dict['fp_due_to_no_overlap_but_high_confidence'])
        recall = cls_score_dict['tp'] / (cls_score_dict['tp'] + cls_score_dict['fn_due_to_low_score_or_wrong_cls'] + cls_score_dict['fn_due_to_no_overlap'])
        logger.info(f'{cls_score_key}_precision: {prec}')
        logger.info(f'{cls_score_key}_recall: {recall}')

    
    # Print train loss
    for cls in classes:
        if train_l_dict is not None:
            ret_dict[f'point_loss_cls_{cls}'] = train_loss_dict[f'point_loss_cls_{cls}'] / max(train_loss_dict[f'point_loss_num_{cls}_pts'], 1)
            ret_dict[f'point_loss_box_{cls}'] = train_loss_dict[f'point_loss_box_{cls}'] / max(train_loss_dict[f'point_loss_num_{cls}_pts'], 1)
            ret_dict[f'rcnn_loss_reg_{cls}'] = train_loss_dict[f'rcnn_loss_reg_{cls}'] / max(train_loss_dict[f'rcnn_loss_num_{cls}_boxes'], 1)
            ret_dict[f'rcnn_loss_corner_{cls}'] = train_loss_dict[f'rcnn_loss_corner_{cls}'] / max(train_loss_dict[f'rcnn_loss_num_{cls}_boxes'], 1)
            ret_dict[f'rcnn_loss_cls_{cls}'] = train_loss_dict[f'rcnn_loss_cls_{cls}'] / max(train_loss_dict[f'rcnn_loss_num_{cls}_rois'], 1)

            loss = ret_dict[f'point_loss_cls_{cls}']
            logger.info(f'point_loss_cls_{cls}: {loss}')
            
            loss = ret_dict[f'point_loss_box_{cls}']
            logger.info(f'point_loss_box_{cls}: {loss}')
            
            loss = ret_dict[f'rcnn_loss_reg_{cls}']
            logger.info(f'rcnn_loss_reg_{cls}: {loss}')
            
            loss = ret_dict[f'rcnn_loss_corner_{cls}']
            logger.info(f'rcnn_loss_corner_{cls}: {loss}')

    # Print test loss
    logger.info('*************** Test Loss *****************')
    for nms_key in [ 'before_nms', 'after_nms']: #'after_nms',
        for key, val in test_loss_dict[nms_key].items(): #, 'pred'
            #Test loss
            loss = np.sum(val) / len(val)
            logger.info(f'{nms_key}_{key}: {loss}') 

    # Print recall
    logger.info('*************** RECALL *****************')
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall


        for cls in classes:
            gt_cls_num = metric[f'gt_{cls}_num']
            cur_roi_recall_cls = metric[f'recall_roi_{cur_thresh}_{cls}' ] / max(gt_cls_num, 1)
            cur_rcnn_recall_cls = metric[f'recall_rcnn_{cur_thresh}_{cls}' ] / max(gt_cls_num, 1)

            logger.info(f'recall_roi_{cur_thresh}_{cls}: {cur_roi_recall_cls}')
            logger.info(f'recall_rcnn_{cur_thresh}_{cls}: {cur_rcnn_recall_cls}')
            logger.info(f'gt_{cls}_num: {gt_cls_num}')
            
            ret_dict[f'recall/roi_{cur_thresh}_{cls}'] = cur_roi_recall_cls
            ret_dict[f'recall/rcnn_{cur_thresh}_{cls}'] = cur_rcnn_recall_cls


    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
        eval_levels_cfg=cfg.MODEL.POST_PROCESSING.get('EVAL_LEVELS', None)
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_pickle(cfg, pickle_file, disard_results, dataloader, result_dir, epoch_id, logger):
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names

    logger.info('Evaluation on pickle file: {}'.format(pickle_file))

    det_annos = []
    with open(pickle_file, 'rb') as f:
        det_annos = pickle.load(f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        eval_levels_list_cfg=cfg.MODEL.POST_PROCESSING.get('EVAL_LEVELS_LIST', None)
    )

    ret_dict = {
        'EVAL_LEVELS_LIST': cfg.MODEL.POST_PROCESSING.get('EVAL_LEVELS_LIST', None)
    }

    logger.info(result_str)
    if isinstance(result_dict, dict):
        ret_dict.update(result_dict)
    else:
        ret_dict['distance'] = result_dict
    logger.info(ret_dict)

    if not disard_results:
        logger.info('Eval is saved to %s' % result_dir)
        with open(result_dir / '{}_eval.pkl'.format(cfg.TAG), 'wb') as f:
            pickle.dump(ret_dict, f)
    else:
        logger.info('Not saving results')

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
