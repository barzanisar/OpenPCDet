import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from third_party.OpenPCDet.pcdet.models.pretext_heads.mlp import MLP
from third_party.OpenPCDet.pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from third_party.OpenPCDet.pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from third_party.OpenPCDet.pcdet.utils.spconv_utils import spconv
from third_party.OpenPCDet.pcdet.models.pretext_heads.gather_utils import *


class SegVoxHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size):
        super().__init__()
        self.use_mlp = False
        self.NUM_KEYPOINTS = model_cfg.num_keypts
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head

        #self.dout=nn.Dropout(p=0.3)

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, batch_dict, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus

        #Not used in the future: voxels, voxel_num_points, encoded_spconv_tensor
        #Used in the future: voxel_coords, points(used is in voxel set abstraction in the past), multi_scale_3d_features (in voxel set abstraction and voxelrcnn head), spatial_features, spatial_features_2d
        # points = batch_dict['points']
        voxel_coords = batch_dict['voxel_coords']
        # multi_scale_3d_features = batch_dict['multi_scale_3d_features']
        spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']
        batch_size_this = batch_dict['batch_size']

        ################################ Unshuffle Points ###########################3
        # num_pts_batch = np.unique(points[:,0].cpu().numpy(), return_counts=True)[1]
        # all_size = concat_all_gather(torch.tensor(num_pts_batch).cuda())
        # max_size = torch.max(all_size) #max num voxels in any pc
        # points_gather = gather_feats(batch_indices=points[:,0], 
        #                                 feats_or_coords=points, 
        #                                 batch_size_this=batch_size_this, 
        #                                 num_vox_or_pts_batch=num_pts_batch, 
        #                                 max_num_vox_or_pts=max_size)

        # batch_size_all = points_gather.shape[0]
        # num_gpus = batch_size_all // batch_size_this
        # # restored index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx] # indices of shuffled pc id [4,6,5,1,.,0,....] -> [[4,6,5,1,.,0,.],[ second 8 indices]] -> pick for this gpu [4,6,5,1,.,0,.,.]

        # batch_dict['batch_size'] = len(idx_this)
        # batch_dict['points'] = get_feats_this(idx_this, all_size, points_gather, is_ind=True) # (N1+..Nbs, bxyzi)
   
        ################################ Unshuffle voxels ###########################3

        # Each pc has diff num voxels at diff layers
        # num voxels in input layer for this batch
        num_voxels_batch = np.unique(voxel_coords[:,0].cpu().numpy(), return_counts=True)[1]
        # num voxels in input layer for all batches
        all_size = concat_all_gather(torch.tensor(num_voxels_batch).cuda()) #[num voxels pc 1, ...., num voxels pc 16]
        max_size = torch.max(all_size) #max num voxels in any pc

        voxel_coords_gather = gather_feats(batch_indices=voxel_coords[:,0], 
                                        feats_or_coords=voxel_coords, 
                                        batch_size_this=batch_size_this, 
                                        num_vox_or_pts_batch=num_voxels_batch, 
                                        max_num_vox_or_pts=max_size)
        
        spatial_features_gather = concat_all_gather(spatial_features)
        spatial_features_2d_gather = concat_all_gather(spatial_features_2d)

        batch_size_all = voxel_coords_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx] # indices of shuffled pc id [4,6,5,1,.,0,....] -> [[4,6,5,1,.,0,.],[ second 8 indices]] -> pick for this gpu [4,6,5,1,.,0,.,.]

        batch_dict['batch_size'] = len(idx_this)

        batch_dict['voxel_coords'] = get_feats_this(idx_this, all_size, voxel_coords_gather, is_ind=True)
        batch_dict['spatial_features'] = spatial_features_gather[idx_this]
        batch_dict['spatial_features_2d'] = spatial_features_2d_gather[idx_this]
        
        ################## This takes a lot of time and is not needed for centerpoint
        # multi_scale_3d_features_this={}

        # for key in multi_scale_3d_features.keys():

        #     vox_feats, vox_ind = multi_scale_3d_features[key].features, multi_scale_3d_features[key].indices

        #     # num voxels in this layer for this batch
        #     num_voxels_batch = np.unique(vox_ind[:,0].cpu().numpy(), return_counts=True)[1]
        #     # num voxels in this layer for all batches
        #     all_size = concat_all_gather(torch.tensor(num_voxels_batch).cuda()) #[num voxels pc 1, ...., num voxels pc 16]
        #     max_size = torch.max(all_size) #max num voxels in any pc

        #     vox_feats_gather = gather_feats(batch_indices=vox_ind[:,0], 
        #                                     feats_or_coords=vox_feats, 
        #                                     batch_size_this=batch_size_this, 
        #                                     num_vox_or_pts_batch=num_voxels_batch, 
        #                                     max_num_vox_or_pts=max_size)
            
        #     vox_ind_gather = gather_feats(batch_indices=vox_ind[:,0], 
        #                                     feats_or_coords=vox_ind, 
        #                                     batch_size_this=batch_size_this, 
        #                                     num_vox_or_pts_batch=num_voxels_batch, 
        #                                     max_num_vox_or_pts=max_size)
            
        #     vox_feats_this = get_feats_this(idx_this, all_size, vox_feats_gather)
        #     vox_ind_this = get_feats_this(idx_this, all_size, vox_ind_gather, is_ind=True)
            
        #     multi_scale_3d_features_this[key] = spconv.SparseConvTensor(
        #     features=vox_feats_this,
        #     indices=vox_ind_this.int(),
        #     spatial_shape=multi_scale_3d_features[key].spatial_shape,
        #     batch_size=batch_dict['batch_size']
        #     )


        # batch_dict['multi_scale_3d_features'] = multi_scale_3d_features_this

        return batch_dict

    def interpolate_from_bev_features(self, keypoints, bev_features, bev_stride):
        """
        Args:
            keypoints: (N 3=xyz)
            bev_features: (C=256, H=188, W=188)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0] # x coord in 1504x1504 grid
        y_idxs = (keypoints[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        # Getting keypoints xy bev grid coord
        x_idxs = x_idxs / bev_stride # x coord in 188x188 BEV grid
        y_idxs = y_idxs / bev_stride

        #bev_features.permute(1, 2, 0)  # (H=188, W=188, C=256)
        point_bev_features = bilinear_interpolate_torch(bev_features.permute(1, 2, 0), x_idxs, y_idxs) #(4096 keypts, bevfeat_dim=256)

        # point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C) (B x 4096, 256)
        return point_bev_features
    
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

        # unshuffle point features, points (not needed), batch size bcz no BN in projection layers
        idx_unshuffle = batch_dict.get('idx_unshuffle', None)
        if torch.distributed.is_initialized() and idx_unshuffle is not None:
            batch_dict = self._batch_unshuffle_ddp(batch_dict, idx_unshuffle)

        
        cluster_ids = batch_dict['cluster_ids'] #original (N1+...+Nbs)
        backbone_3d_bev_feats = batch_dict['spatial_features'] # (6,256,188,188) # unshuffled (B=2, C=128, N num points = 20000)
        backbone_2d_bev_feats = batch_dict['spatial_features_2d'] # (6,512,188,188)
        bev_features = torch.cat([backbone_3d_bev_feats, backbone_2d_bev_feats], dim=1)  # (6,256+512,188,188)
        points = batch_dict['points'] #(N, bxyzi)
        batch_size = batch_dict["batch_size"] # unshuffled

        batch_seg_feats = []
        for pc_idx in range(batch_size):
            b_mask = points[:,0] == pc_idx
            pc = points[b_mask][:,1:4]
            cluster_labels_this_pc = cluster_ids[pc_idx] # (Num pts this pc,)
            common_cluster_labels_this_pc = batch_dict['common_cluster_ids'][pc_idx]
            fg_pts_mask = cluster_labels_this_pc > -1
            fg_points = pc[fg_pts_mask][:,0:3].unsqueeze(dim=0) #(1, num fg pts, 3=xyz)

            # Sample fg points
            cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    fg_points[:, :, 0:3].contiguous(), self.NUM_KEYPOINTS
                ).long()# (1, 20k) sample 20k points from this pc
            

            if fg_points.shape[1] < self.NUM_KEYPOINTS:
                times = int(self.NUM_KEYPOINTS / fg_points.shape[1]) + 1
                non_empty = cur_pt_idxs[0, :fg_points.shape[1]]
                cur_pt_idxs[0] = non_empty.repeat(times)[:self.NUM_KEYPOINTS]

            sampled_fg_keypoints = fg_points[0][cur_pt_idxs[0]] #(Num pts, 3=xyz)
            sampled_fg_cluster_lbls = cluster_labels_this_pc[fg_pts_mask][cur_pt_idxs[0].cpu()] #(Num pts, )
            
            #interpolate from bev features at the FPS sampled fg keypts
            keypoint_bev_features = self.interpolate_from_bev_features(
                sampled_fg_keypoints, bev_features[pc_idx],
                bev_stride=batch_dict['spatial_features_stride']
            ) #(num keypts, 512+256=768)
            
            for segment_lbl in common_cluster_labels_this_pc: #np.unique(cluster_labels_this_pc):
                assert segment_lbl != -1

                seg_feats = keypoint_bev_features[sampled_fg_cluster_lbls==segment_lbl,:] #(npts this seg, 768)

                seg_feats = seg_feats.permute(1,0).unsqueeze(0) #1, 768, npts this seg
                npoints = seg_feats.shape[-1]
                seg_max_feat = F.max_pool1d(seg_feats, npoints).squeeze(-1) #[1, 768, npoints] -> [1, 768, 1] -> [1, 768]
                batch_seg_feats.append(seg_max_feat) # (1, 768)

        all_seg_feats = torch.vstack(batch_seg_feats) # (num clusters, 768)
        if self.use_mlp:
            all_seg_feats = self.head(all_seg_feats) # (num clusters, 128)       


        batch_dict["pretext_head_feats"] = all_seg_feats # (num clusters, 128)

        return batch_dict        