import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from third_party.OpenPCDet.pcdet.utils import loss_utils
def make_fc_layers(fc_cfg, input_channels, output_channels):
    fc_layers = []
    c_in = input_channels
    for k in range(0, fc_cfg.__len__()):
        fc_layers.extend([
            nn.Linear(c_in, fc_cfg[k], bias=False),
            nn.BatchNorm1d(fc_cfg[k]),
            nn.ReLU(),
        ])
        c_in = fc_cfg[k]
    fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
    return nn.Sequential(*fc_layers)

class RegHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size):
        super().__init__()
        self.loss_weight = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seg_reg_weight']
        self.reg_fc_layers = make_fc_layers(
            fc_cfg=model_cfg.REG_FC,
            input_channels=256,
            output_channels=3
        ) #[256+bn+relu, 256+bn+relu, 3=cos(r1-r2), sin(r1-r2), s1/s2]

        reg_loss_type = model_cfg.LOSS_CONFIG.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                point_features: (B, C, N)
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = int(batch_dict['gt_boxes'].shape[0]/2)
        seg_feats = batch_dict['pretext_head_feats'] # (num common clusters, 128+128)
        num_segs = seg_feats.shape[0]

        delta_rot_scale_ratio_preds = self.reg_fc_layers(seg_feats) #(num common clusters, 128+128) -> (num common clusters, 2)
        gt_boxes = batch_dict['gt_boxes'][0:batch_size]
        gt_boxes_moco = batch_dict['gt_boxes'][batch_size:]
        seg_labels = gt_boxes.new_zeros((num_segs, 3))

        start_seg_idx = 0
        for k in range(batch_size):
            gt_boxes_this_pc = gt_boxes[k, batch_dict['common_cluster_gtbox_idx'][k],:] # (num common clusters, 7) i.e. xyz, lwh, rz
            gt_boxes_moco_this_pc = gt_boxes_moco[k, batch_dict['common_cluster_gtbox_idx_moco'][k],:] # (num common clusters, 7) i.e. xyz, lwh, rz
            num_boxes = gt_boxes_this_pc.shape[0]

            delta_rot = gt_boxes_this_pc[:,6] - gt_boxes_moco_this_pc[:,6]
            seg_labels[start_seg_idx:start_seg_idx+num_boxes,0] = torch.cos(delta_rot)
            seg_labels[start_seg_idx:start_seg_idx+num_boxes,1] = torch.sin(delta_rot)
            seg_labels[start_seg_idx:start_seg_idx+num_boxes,2] = gt_boxes_this_pc[:,3] / gt_boxes_moco_this_pc[:,3]

            start_seg_idx+=num_boxes

        seg_reg_loss_src = self.reg_loss_func(
            delta_rot_scale_ratio_preds[None, ...], seg_labels[None, ...]
        )# (1, 16384x2, 8)

        normalizer = max(float(num_segs), 1.0) #torch.clamp(num_segs, min=1.0)

        loss_sum = seg_reg_loss_src.sum(dim=1)/normalizer #(1, rot loss on cos(delta_rot), rot loss on sin(delta_rot), scale loss)
        batch_dict['seg_reg_loss_rot'] = (loss_sum[0, 0] + loss_sum[0, 1]).item() 
        batch_dict['seg_reg_loss_scale'] = loss_sum[0, 2].item()

        seg_reg_loss = self.loss_weight * loss_sum.sum() 


        batch_dict['seg_reg_loss'] = seg_reg_loss

        return batch_dict        