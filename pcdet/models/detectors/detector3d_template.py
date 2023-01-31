import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils


class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels=model_info_dict['backbone_channels'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Perform NMS on RCNN box predictions to remove overlapping predicted boxes (100 boxes in one pc shortlisted to e.g. 4 boxes)

        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:
            recall_dict:
                gt: (M) total num of gt boxes in this batch i.e. 2 pcs
                roi_0.3: (k) num of gt boxes that overlap/matched with predicted rois (from 1st stage i.e. (100 boxes, 7)) with iou3d > 0.3 (where k <=M)
                roi_0.5: (m) num of gt boxes that overlap with predicted rois (from 1st stage) with iou3d > 0.5 (where m <=M)
                roi_0.7: (n) num of gt boxes that overlap with predicted rois (from 1st stage) with iou3d > 0.7 (where n <=M) (hardest! so very few boxes)
                k >= m >= n, i.e. k includes m and n, m includes n

                rcnn_0.3: (a) num of gt boxes that overlap with final predicted rcnn boxes (i.e. (100 boxes, 7) before nms) with iou3d > 0.3 (where a <=M)
                rcnn_0.5: (b) num of gt boxes that overlap with final predicted rcnn boxes with iou3d > 0.5 (where b <=M)
                rcnn_0.7: (c) num of gt boxes that overlap with final predicted rcnn boxes before nms) with iou3d > 0.7 (where c <=M) (hardest! so very few boxes)
                a >= b >= c, i.e. a includes b and c, b includes c
                We expect rcnn numbers to be higher than roi numbers bcz rcnn output is 2nd stage more refined output

            pred_dict: final output after NMS (a list of size 2 = batch size)
                pred_labels: # example shape: (4), predicted 1st stage roi_labels for nms shortlisted boxes [1: car, 2: ped, 3: cyc]
                pred_scores: # example shape: (4), RCNN predicted box objectness probabilities for nms shortlisted boxes
                pred_boxes:  # example shape: (4), RCNN predicted boxes for nms shortlisted boxes
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        loss_dict = {'before_nms': {}, 'after_nms':{}}
        pred_dicts = []
        conf_matrix = {}
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask] #(100, 7) from rcnn output: final box predictions
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask] #(100, 1) from rcnn output: box objectness scores

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds) # turn rcnn box objectness scores to probabilities
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1) # (100) same as cls_preds, (100) all zeros
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index] # roi_labels: (100) predicted class labels from 1st stage [1: car, 2: ped, 3: cyc]
                else:
                    label_preds = label_preds + 1
                
                # Perform NMS on RCNN box predictions to remove overlapping predicted boxes (100 boxes shortlisted to 4 boxes)
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                ) # shortlisted rcnn predicted box indices and cls_preds

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores # example shape: (4), RCNN predicted box objectness prob for nms shortlisted boxes
                final_labels = label_preds[selected]  # example shape: (4), predicted 1st stage roi_labels for nms shortlisted boxes [1: car, 2: ped, 3: cyc]
                final_boxes = box_preds[selected]  # example shape: (4), RCNN predicted boxes for nms shortlisted boxes
                final_cls_scores = batch_dict['roi_cls_scores'][index][selected]  # example shape: (4, 3), predicted 1st stage roi_cls_scores for nms shortlisted boxes
            
            # Compute recall on rcnn pred boxes and 1st stage rois before NMS!
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            ) # 'gt': total num of gt boxes

            #Before NMS regression and classification loss
            loss_dict['before_nms'] = self.reg_cls_box_loss(
                gt_boxes=batch_dict['gt_boxes'][index], 
                pred_boxes=src_box_preds, 
                pred_box_object_score= src_cls_preds, 
                pred_box_cls_score=batch_dict['roi_cls_scores'][index], 
                loss_dict=loss_dict['before_nms'])


            #After NMS conf matrix for different objectness score thresholds
            conf_matrix = self.compute_conf_matrix(
                gt_boxes=batch_dict['gt_boxes'][index], 
                pred_boxes=final_boxes, 
                pred_box_object_prob= final_scores, 
                pred_box_cls_labels=final_labels, 
                conf_matrix=conf_matrix)

            # #Before NMS conf matrix for different objectness score thresholds
            # conf_matrix = self.compute_conf_matrix(
            #     gt_boxes=batch_dict['gt_boxes'][index], 
            #     pred_boxes=src_box_preds, 
            #     pred_box_object_score= src_cls_preds, 
            #     pred_box_cls_labels=batch_dict['roi_cls_scores'][index].max(dim=1)[1] + 1, #1:car, 2:ped, 3:cyc
            #     conf_matrix=conf_matrix)

            

            # After NMS regression and classification loss
            loss_dict['after_nms'] = self.reg_cls_box_loss(
                gt_boxes=batch_dict['gt_boxes'][index], 
                pred_boxes=final_boxes, 
                pred_box_object_score=final_scores, 
                pred_box_cls_score=final_cls_scores, 
                loss_dict=loss_dict['after_nms'])

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict, loss_dict, conf_matrix

    def compute_conf_matrix(self, gt_boxes, pred_boxes, pred_box_object_prob, pred_box_cls_labels, conf_matrix):
        classes = ['car', 'ped', 'cyc']
        score_threshs = [0.1, 0.5, 0.8]

        if conf_matrix.__len__() == 0:
            for cls in classes:
                for score in score_threshs:
                    conf_matrix[f'{cls}_{score}'] = {'tp': 0, 'fp_due_to_wrong_cls': 0, 'fp_due_to_no_overlap_but_high_confidence': 0,
                    'fn_due_to_low_score_or_wrong_cls': 0, 
                    'fn_due_to_no_overlap': 0, 
                    'fn_due_to_low_score_and_wrong_cls': 0,
                    'fn_due_to_low_score': 0,
                    'fn_due_to_wrong_cls': 0}
        
        # Select non zero gt boxes
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        iou_thresh = 0.25
        if cur_gt.shape[0] > 0:
            if pred_boxes.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, 0:7], cur_gt[:, 0:7])
                for score_thresh in score_threshs:
                    gt_box_wise_matched_scores, gt_box_wise_idx_of_pred_box_max = iou3d_rcnn.max(dim=0) #(num gt boxes)
                    gt_matched_mask = (gt_box_wise_matched_scores > iou_thresh)  #(num gt boxes), true if matched i.e. iou > 0.1
                    pred_box_matched_idx = gt_box_wise_idx_of_pred_box_max[gt_matched_mask]  #(num matched gt boxes)
                    
                    gt_matched_cls_label = cur_gt[gt_matched_mask, -1]  #(num matched gt boxes)
                    pred_box_matched_cls_label = pred_box_cls_labels[pred_box_matched_idx] #(num matched gt boxes) 1:car, 2:ped: 3:cyc
                    pred_box_matched_probs = pred_box_object_prob[pred_box_matched_idx] #(num matched gt boxes)

                    # TP: if objectness scores > score_thresh and labels match
                    tp_mask = (pred_box_matched_probs >= score_thresh) & (gt_matched_cls_label == pred_box_matched_cls_label)#(num matched gt boxes)
                    tp_labels = gt_matched_cls_label[tp_mask] #(num tps)
                    
                    # FN in matched gt boxes: 
                    fn_mask_due_to_low_score_or_wrong_cls = (pred_box_matched_probs < score_thresh) | (gt_matched_cls_label != pred_box_matched_cls_label)
                    fn_labels_due_to_low_score_or_wrong_cls = gt_matched_cls_label[fn_mask_due_to_low_score_or_wrong_cls]

                    fn_mask_due_to_low_score_and_wrong_cls = (pred_box_matched_probs < score_thresh) & (gt_matched_cls_label != pred_box_matched_cls_label)
                    fn_labels_due_to_low_score_and_wrong_cls = gt_matched_cls_label[fn_mask_due_to_low_score_and_wrong_cls]

                    fn_mask_due_to_low_score = pred_box_matched_probs < score_thresh
                    fn_labels_due_to_low_score = gt_matched_cls_label[fn_mask_due_to_low_score]

                    wrong_cls_matching_mask = gt_matched_cls_label != pred_box_matched_cls_label
                    fn_labels_due_to_wrong_cls = gt_matched_cls_label[wrong_cls_matching_mask]
                    fp_labels_due_to_wrong_cls = pred_box_matched_cls_label[wrong_cls_matching_mask] # predicted labels for wrong class matching
                    fp_mask_due_to_wrong_cls_but_high_score = pred_box_matched_probs[wrong_cls_matching_mask] > score_thresh  
                    # FN in unmatched gt boxes:
                    gt_unmatched_cls_label = cur_gt[gt_matched_mask==0, -1]  #(num unmatched gt boxes)

                    # FP in unmatched pred boxes
                    pred_box_unmatched_mask = gt_matched_mask.new_ones((pred_boxes.shape[0]))
                    pred_box_unmatched_mask[pred_box_matched_idx] = 0
                    pred_box_unmatched_pred_labels =  pred_box_cls_labels[pred_box_unmatched_mask]
                    fp_unmatched_but_high_conf_mask = pred_box_object_prob[pred_box_unmatched_mask] > score_thresh

                    for i, cls in enumerate(classes):
                        class_id = i+1
                        # Add car, ped, cyc tps: num gt boxes matched bcz max overlapping pred boxes has both score > thresh and correct class
                        conf_matrix[f'{cls}_{score_thresh}']['tp'] += (tp_labels == class_id).sum().item() 
                        
                        # Add FNs due to either low score or wrong cls: num gt boxes unmatched bcz max overlapping pred boxes either has low score or wrong cls
                        conf_matrix[f'{cls}_{score_thresh}']['fn_due_to_low_score_or_wrong_cls'] += (fn_labels_due_to_low_score_or_wrong_cls == class_id).sum().item() 
                        conf_matrix[f'{cls}_{score_thresh}']['fn_due_to_low_score_and_wrong_cls'] += (fn_labels_due_to_low_score_and_wrong_cls == class_id).sum().item() 
                        conf_matrix[f'{cls}_{score_thresh}']['fn_due_to_low_score'] += (fn_labels_due_to_low_score == class_id).sum().item() 
                        conf_matrix[f'{cls}_{score_thresh}']['fn_due_to_wrong_cls'] += (fn_labels_due_to_wrong_cls == class_id).sum().item() 

                        # Add FPs: num pred boxes overlapping/matching with gt box of wrong class
                        conf_matrix[f'{cls}_{score_thresh}']['fp_due_to_wrong_cls'] += ((fp_labels_due_to_wrong_cls == class_id) & (fp_mask_due_to_wrong_cls_but_high_score)).sum().item() 

                        # Add FNs due to no overlap: num gt boxes unmatched due to no overlap
                        conf_matrix[f'{cls}_{score_thresh}']['fn_due_to_no_overlap'] += (gt_unmatched_cls_label == class_id).sum().item()
                        
                        # Add FPs: num pred boxes not overlapping with any gt box but with objectness score > thresh
                        conf_matrix[f'{cls}_{score_thresh}']['fp_due_to_no_overlap_but_high_confidence'] += ((pred_box_unmatched_pred_labels == class_id) & (fp_unmatched_but_high_conf_mask)).sum().item()

                        
        return conf_matrix

    def reg_cls_box_loss(self, gt_boxes, pred_boxes, pred_box_object_score, pred_box_cls_score, loss_dict):
        classes = ['car', 'ped', 'cyc']
        if loss_dict.__len__() == 0:
            for cls in classes:
                loss_dict[f'gt_wise_cls_loss_{cls}'] = 0
                loss_dict[f'gt_wise_objectness_loss_{cls}'] = 0
                loss_dict[f'gt_wise_box_position_loss_{cls}'] = 0
                loss_dict[f'gt_wise_box_dim_loss_{cls}'] = 0

                # loss_dict[f'pred_wise_cls_loss_{cls}'] = 0
                # loss_dict[f'pred_wise_objectness_loss_{cls}'] = 0
                # loss_dict[f'pred_wise_box_position_loss_{cls}'] = 0
                # loss_dict[f'pred_wise_box_dim_loss_{cls}'] = 0

        # Select non zero gt boxes
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        iou_thresh = 0.25
        if cur_gt.shape[0] > 0:
            if pred_boxes.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, 0:7], cur_gt[:, 0:7])
                for i, cls in enumerate(classes):
                    class_id = i+1
                    gt_box_wise_matched_scores, gt_box_wise_idx_of_pred_box_matched = iou3d_rcnn.max(dim=0) #(num gt boxes)
                    gt_cls_matched_mask = ( (gt_box_wise_matched_scores > iou_thresh) & (cur_gt[:,-1] == class_id) )
                    matched_gt_cls_boxes = cur_gt[gt_cls_matched_mask]
                    gt_cls_boxes_matched_num = matched_gt_cls_boxes.shape[0]

                    if gt_cls_boxes_matched_num > 0:
                        idx_of_pred_box_matched = gt_box_wise_idx_of_pred_box_matched[gt_cls_matched_mask]
                        matched_pred_cls_boxes =  pred_boxes[idx_of_pred_box_matched]
                        pred_box_object_score_matched = pred_box_object_score[idx_of_pred_box_matched]
                        pred_box_cls_score_matched = pred_box_cls_score[idx_of_pred_box_matched]


                        mse_pos = torch.linalg.norm(matched_gt_cls_boxes[:, 0:3] - matched_pred_cls_boxes[:,0:3], dim = 1).sum()/gt_cls_boxes_matched_num
                        mse_lens = torch.linalg.norm(matched_gt_cls_boxes[:, 3:6] - matched_pred_cls_boxes[:,3:6], dim = 1).sum()/gt_cls_boxes_matched_num

                        exp_scores = torch.exp(pred_box_cls_score_matched)
                        exp_scores_sum = exp_scores.sum(dim=1)
                        softmax_prob = exp_scores / exp_scores_sum.view(-1,1)
                        
                        loss_cls_objectness = -torch.log(torch.sigmoid(pred_box_object_score_matched)).sum()/gt_cls_boxes_matched_num
                        loss_cls_classif = -torch.log(softmax_prob[:,i]).sum()/gt_cls_boxes_matched_num

                        loss_dict[f'gt_wise_cls_loss_{cls}'] += loss_cls_classif.item()
                        loss_dict[f'gt_wise_objectness_loss_{cls}'] += loss_cls_objectness.item()
                        loss_dict[f'gt_wise_box_position_loss_{cls}'] += mse_pos.item()
                        loss_dict[f'gt_wise_box_dim_loss_{cls}'] += mse_lens.item()
                
                # for i, cls in enumerate(classes):
                #     class_id = i+1
                #     pred_box_wise_matched_scores, pred_box_wise_idx_of_gt_box_matched = iou3d_rcnn.max(dim=1)
                #     pred_box_cls_matched_mask = ( (pred_box_wise_matched_scores > iou_thresh) & (cur_gt[pred_box_wise_idx_of_gt_box_matched,-1] == class_id) )
                #     matched_pred_cls_boxes =  pred_boxes[pred_box_cls_matched_mask]
                #     pred_cls_boxes_matched_num = matched_pred_cls_boxes.shape[0]

                #     if pred_cls_boxes_matched_num > 0:
                #         idx_of_gt_box_matched = pred_box_wise_idx_of_gt_box_matched[pred_box_cls_matched_mask]
                #         matched_gt_cls_boxes = cur_gt[idx_of_gt_box_matched]
                #         pred_box_object_score_matched = pred_box_object_score[pred_box_cls_matched_mask]
                #         pred_box_cls_score_matched = pred_box_cls_score[pred_box_cls_matched_mask]


                #         mse_pos = torch.linalg.norm(matched_gt_cls_boxes[:, 0:3] - matched_pred_cls_boxes[:,0:3], dim = 1).sum()/pred_cls_boxes_matched_num
                #         mse_lens = torch.linalg.norm(matched_gt_cls_boxes[:, 3:6] - matched_pred_cls_boxes[:,3:6], dim = 1).sum()/pred_cls_boxes_matched_num

                #         exp_scores = torch.exp(pred_box_cls_score_matched)
                #         exp_scores_sum = exp_scores.sum(dim=1)
                #         softmax_prob = exp_scores / exp_scores_sum.view(-1,1)

                #         loss_cls_objectness = -torch.log(torch.sigmoid(pred_box_object_score_matched)).sum()/pred_cls_boxes_matched_num
                #         loss_cls_classif = -torch.log(softmax_prob[:,i]).sum()/pred_cls_boxes_matched_num

                #         loss_dict[f'pred_wise_cls_loss_{cls}'] += loss_cls_classif.item()
                #         loss_dict[f'pred_wise_objectness_loss_{cls}'] += loss_cls_objectness.item()
                #         loss_dict[f'pred_wise_box_position_loss_{cls}'] += mse_pos.item()
                #         loss_dict[f'pred_wise_box_dim_loss_{cls}'] += mse_lens.item()
        
        return loss_dict
    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        # box_preds: #(100, 7) from rcnn output: final box predictions
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None # predicted rois from 1st stage: (100, 7)
        gt_boxes = data_dict['gt_boxes'][batch_index] #(M, 8)

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0, 
            'num_car_gt': 0,
            'num_ped_gt': 0,
            'num_cyc_gt': 0 }
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

                # Class wise recall
                recall_dict[f'roi_{cur_thresh}_car'] = 0
                recall_dict[f'roi_{cur_thresh}_ped'] = 0
                recall_dict[f'roi_{cur_thresh}_cyc'] = 0
                recall_dict[f'rcnn_{cur_thresh}_car'] = 0
                recall_dict[f'rcnn_{cur_thresh}_ped'] = 0
                recall_dict[f'rcnn_{cur_thresh}_cyc'] = 0

        # Select non zero gt boxes
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7]) # iou3d matrix of shape (100=rcnn predicted boxes, M=gt boxes)
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7]) # iou3d matrix of shape (100= predicted 1st stage rois, M=gt boxes)

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item() #number of gt boxes that matched/overlapped with rcnn predicted boxes with iou3D > 0.3, 0.5, 0.7
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                    
                    # Class wise rcnn recall
                    rcnn_recalled_car = ( (iou3d_rcnn.max(dim=0)[0] > cur_thresh) & (cur_gt[:,-1] == 1) ).sum().item() #number of gt boxes that matched and are also car
                    rcnn_recalled_ped = ( (iou3d_rcnn.max(dim=0)[0] > cur_thresh) & (cur_gt[:,-1] == 2) ).sum().item() #number of gt boxes that matched and are also car
                    rcnn_recalled_cyc = ( (iou3d_rcnn.max(dim=0)[0] > cur_thresh) & (cur_gt[:,-1] == 3) ).sum().item() #number of gt boxes that matched and are also car
                    recall_dict[f'rcnn_{cur_thresh}_car'] += rcnn_recalled_car
                    recall_dict[f'rcnn_{cur_thresh}_ped'] += rcnn_recalled_ped
                    recall_dict[f'rcnn_{cur_thresh}_cyc'] += rcnn_recalled_cyc

                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item() #number of gt boxes that matched/overlapped with 1st stage roi predicted boxes with iou3D > 0.3, 0.5, 0.7
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled 

                    # Class wise roi recall
                    roi_recalled_car = ( (iou3d_roi.max(dim=0)[0] > cur_thresh) & (cur_gt[:,-1] == 1) ).sum().item()
                    roi_recalled_ped = ( (iou3d_roi.max(dim=0)[0] > cur_thresh) & (cur_gt[:,-1] == 2) ).sum().item()
                    roi_recalled_cyc = ( (iou3d_roi.max(dim=0)[0] > cur_thresh) & (cur_gt[:,-1] == 3) ).sum().item()
                    recall_dict[f'roi_{cur_thresh}_car'] += roi_recalled_car
                    recall_dict[f'roi_{cur_thresh}_ped'] += roi_recalled_ped
                    recall_dict[f'roi_{cur_thresh}_cyc'] += roi_recalled_cyc



            recall_dict['gt'] += cur_gt.shape[0] # total num of gt boxes in this pc
            recall_dict['num_car_gt'] += (cur_gt[:,-1] == 1).sum().item() # total num of gt boxes in this pc
            recall_dict['num_ped_gt'] += (cur_gt[:,-1] == 2).sum().item() # total num of gt boxes in this pc
            recall_dict['num_cyc_gt'] += (cur_gt[:,-1] == 3).sum().item() # total num of gt boxes in this pc
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
