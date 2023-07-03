import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pcdet.models.pretext_heads.mlp import MLP
from pcdet.ops.roipoint_pool3d import roipoint_pool3d_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class BboxHead(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.use_mlp = False
        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head layer 1: 128 in , 128 out, relu. layer  2: 128 in, 128 out 
        
        # self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
        #     num_sampled_points=512,
        #     pool_extra_width=[0.0, 0.0 ,0.0]
        # )


        # self.num_in_channels = 3 + 1 + model_cfg.mlp_dim[0] # xyz + point_depth
        # point_feature_mlps = [self.num_in_channels] + model_cfg.mlp_dim[1:] # 132, 128, 128
        # shared_mlps = []
        # use_bn=False
        # for k in range(len(point_feature_mlps) - 1):
        #     shared_mlps.append(nn.Conv2d(point_feature_mlps[k], point_feature_mlps[k + 1], kernel_size=1, bias=not use_bn))
        #     if use_bn:
        #         shared_mlps.append(nn.BatchNorm2d(point_feature_mlps[k + 1]))
        #     shared_mlps.append(nn.ReLU())
        # self.point_proj_layer = nn.Sequential(*shared_mlps)
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        
        num_gpus = batch_size_all // batch_size_this
        
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def roipool3d_gpu(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C): (2, 128, 7)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C): (16384 x 2, 128)
        Returns:
            pooled_features: (2x128=num rois, 512, 133=[3 = (points (in roi) xyz - roi_center) in roi frame, 1 point cls score, 1 depth, 128 features from pointnet++])
            For empty rois, pooled featrues are filled with zeros
        """
        batch_size = batch_dict['batch_size']
        #batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        rois = batch_dict['gt_boxes']  # (B, num_rois=128, 7 + C)
        # batch_cnt = point_coords.new_zeros(batch_size).int()
        # for bs_idx in range(batch_size):
        #     batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        # assert batch_cnt.min() == batch_cnt.max()

        point_depths = point_coords.norm(dim=1) / 70 - 0.5
        point_features_list = [point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1) #(16384x2, 130) 
        batch_points = point_coords.view(batch_size, -1, 3) #(2, 16384, 3)
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1]) #(2, 16384, 130)

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois=128, num_sampled_points in each roi=512, 133 = 3 + (C=130)), pooled_empty_flag: (B=2, num_rois=128)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2) # pooled features[:, :, :, 0:3] contains vector from roi center to points in roi i.e. (points xyz - roi_center) in lidar frame

            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1]) # (2x128 num rois, 512 points, 133 feature dim)
            
            # Transform pooled features[:, :, :, 0:3] so that it contains (points xyz - roi_center) in roi frame
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_features
    
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

        point_features = batch_dict['point_features'] #(B=8, C=128, N num points = 16384)    
        batch_size = batch_dict["batch_size"]

        idx_unshuffle = batch_dict.get('idx_unshuffle', None)
        if torch.distributed.is_initialized() and idx_unshuffle is not None:
            point_features = self._batch_unshuffle_ddp(point_features, idx_unshuffle)

        # # Method 1
        # pooled_features = self.roipool3d_gpu(batch_dict) #(B=8xnum gt boxes, N=512 points, C=132) xyz from roi center to point, depth, 128 feature vector
        # pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (B, C=132, N=512)

        #TODO: try average pool

        # # Optional: Transform each point feature with 2 mlps
        # feat_input = pooled_features.transpose(1, 2).unsqueeze(dim=3).contiguous()  # (B rois, 132 features, 512 points, 1)
        # pooled_features = self.point_proj_layer(feat_input) # input: (B=256, C=5, 512=H, 1=W) -> Conv2d(5,128, kernel size = (1,1)) + Relu -> Cpnv2d(128,128, kernel size = (1,1)) + relu -> output:(B=256, C=128, 512=H, 1=W) 
        # pooled_features = pooled_features.squeeze(-1)

        # # Max pool
        # seg_max_feat = F.max_pool1d(pooled_features, pooled_features.shape[-1]).squeeze(-1) #[B = 8 x num gt boxes, 132, npoints] -> [B, 132, 1] -> [B, 132]
        
        # if self.use_mlp:
        #     all_seg_feats = self.head(seg_max_feat) # num clusters x 128

        #### Method 2 Start
        batch_seg_feats = []
        total_num_obj=0
        for pc_idx in range(batch_size):
            pc_feats = point_features[pc_idx] #(128 x 16384)
            points_idx = batch_dict['points'][:,0] == pc_idx
            points = batch_dict['points'][points_idx, 1:4]

            gt_boxes = batch_dict["gt_boxes"][pc_idx]

            # Select non zero gt boxes
            k = gt_boxes.__len__() - 1
            while k >= 0 and gt_boxes[k].sum() == 0:
                k -= 1
            nonzero_gt_boxes = gt_boxes[:k + 1]


            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points[:, 0:3].unsqueeze(dim=0),
                nonzero_gt_boxes[:, 0:7].unsqueeze(dim=0)
            ).long().squeeze(dim=0).cpu().numpy()

            num_obj = nonzero_gt_boxes.shape[0]
            total_num_obj += num_obj

            for i in range(num_obj):
                seg_feats = pc_feats[:, box_idxs_of_pts == i] #(128, npoints in this seg)

                #seg_feats = self.dout(seg_feats) #zero some values in [128, num points in this cluster]
                seg_feats = seg_feats.unsqueeze(0) #1,128, npoints in this seg
                npoints = seg_feats.shape[-1]
                seg_max_feat = F.max_pool1d(seg_feats, npoints).squeeze(-1) #[1, 128, npoints] -> [1, 128, 1] -> [1, 128]
                batch_seg_feats.append(seg_max_feat)
        
        all_seg_feats = torch.vstack(batch_seg_feats) # num clusters x 128
        assert all_seg_feats.shape[0] == total_num_obj
        if self.use_mlp:
            all_seg_feats = self.head(all_seg_feats) # num clusters x 128
        #### Method 2 End

        # Convert point features in OpenPCDet format
        batch_dict["pretext_head_feats"] = all_seg_feats #(B=num clusters, C=128)
        point_features = point_features.permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1]) # (16384x2, 128)

        return batch_dict        