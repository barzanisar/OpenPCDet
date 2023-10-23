import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from third_party.OpenPCDet.pcdet.models.pretext_heads.mlp import MLP
from third_party.OpenPCDet.pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from third_party.OpenPCDet.pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from third_party.OpenPCDet.pcdet.utils.spconv_utils import spconv



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
    def gather_feats(self, vox_ind, vox_feats_or_ind, batch_size_this, num_voxels_batch, max_num_vox):
        shuffle_feats=[]
        shuffle_feats_shape = list(vox_feats_or_ind.size())
        shuffle_feats_shape[0] = max_num_vox.item()

        for bidx in range(batch_size_this):
            num_vox_this_pc = num_voxels_batch[bidx]
            b_mask = vox_ind == bidx

            shuffle_feats.append(torch.ones(shuffle_feats_shape).cuda())
            shuffle_feats[bidx][:num_vox_this_pc] =  vox_feats_or_ind[b_mask]


        shuffle_feats = torch.stack(shuffle_feats) #(bs_this, max num vox, C)
        feats_gather = concat_all_gather(shuffle_feats)

        return feats_gather

    @torch.no_grad()
    def get_feats_this(self, idx_this, all_size, gather_feats, is_ind=False):
        feats_this_batch = []

        # after shuffling we get only the actual information of each tensor
        # :actual_size is the information, actual_size:biggest_size are just ones (ignore)
        for idx in range(len(idx_this)):
            pc_idx = idx_this[idx]
            num_vox_this_pc = all_size[idx_this[idx]]
            feats_this_pc = gather_feats[pc_idx][:num_vox_this_pc]
            if is_ind:
                feats_this_pc[:,0] = idx #change b_id in vox coords to new bid
            feats_this_batch.append(feats_this_pc) #(num voxels for pc idx, 4=bid, zyx)
        
        feats_this_batch = torch.cat(feats_this_batch, dim=0)
        return feats_this_batch


    @torch.no_grad()
    def _batch_unshuffle_ddp(self, batch_dict, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus

        #Not used in the future: voxels, voxel_num_points, encoded_spconv_tensor
        #Used in the future: voxel_coords, points(used in in voxel set abstraction), multi_scale_3d_features (in voxel set abstraction and voxelrcnn head), spatial_features, spatial_features_2d

        voxel_coords = batch_dict['voxel_coords']
        #points = batch_dict['points']
        multi_scale_3d_features = batch_dict['multi_scale_3d_features']
        spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']
        batch_size_this = batch_dict['batch_size']



        # Each pc has diff num voxels at diff layers
        # num voxels in input layer for this batch
        num_voxels_batch = np.unique(voxel_coords[:,0].cpu().numpy(), return_counts=True)[1]
        # num voxels in input layer for all batches
        all_size = concat_all_gather(torch.tensor(num_voxels_batch).cuda()) #[num voxels pc 1, ...., num voxels pc 16]
        max_size = torch.max(all_size) #max num voxels in any pc

        voxel_coords_gather = self.gather_feats(vox_ind=voxel_coords[:,0], 
                                        vox_feats_or_ind=voxel_coords, 
                                        batch_size_this=batch_size_this, 
                                        num_voxels_batch=num_voxels_batch, 
                                        max_num_vox=max_size)
        
        spatial_features_gather = concat_all_gather(spatial_features)
        spatial_features_2d_gather = concat_all_gather(spatial_features_2d)


        batch_size_all = voxel_coords_gather.shape[0] # 12 if 2 gpus bcz bs=6 is per gpu

        num_gpus = batch_size_all // batch_size_this # 12/6=2
        
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx] # indices of shuffled pc id [4,6,5,1,.,0,....] -> [[4,6,5,1,.,0,.],[ second 8 indices]] -> pick for this gpu [4,6,5,1,.,0,.,.]

        batch_dict['voxel_coords'] = self.get_feats_this(idx_this, all_size, voxel_coords_gather, is_ind=True)
        batch_dict['batch_size'] = len(idx_this)
        batch_dict['spatial_features'] = spatial_features_gather[idx_this]
        batch_dict['spatial_features_2d'] = spatial_features_2d_gather[idx_this]
        
        multi_scale_3d_features_this={}

        for key in multi_scale_3d_features.keys():

            vox_feats, vox_ind = multi_scale_3d_features[key].features, multi_scale_3d_features[key].indices

            # num voxels in this layer for this batch
            num_voxels_batch = np.unique(vox_ind[:,0].cpu().numpy(), return_counts=True)[1]
            # num voxels in this layer for all batches
            all_size = concat_all_gather(torch.tensor(num_voxels_batch).cuda()) #[num voxels pc 1, ...., num voxels pc 16]
            max_size = torch.max(all_size) #max num voxels in any pc

            vox_feats_gather = self.gather_feats(vox_ind=vox_ind[:,0], 
                                            vox_feats_or_ind=vox_feats, 
                                            batch_size_this=batch_size_this, 
                                            num_voxels_batch=num_voxels_batch, 
                                            max_num_vox=max_size)
            
            vox_ind_gather = self.gather_feats(vox_ind=vox_ind[:,0], 
                                            vox_feats_or_ind=vox_ind, 
                                            batch_size_this=batch_size_this, 
                                            num_voxels_batch=num_voxels_batch, 
                                            max_num_vox=max_size)
            
            vox_feats_this = self.get_feats_this(idx_this, all_size, vox_feats_gather)
            vox_ind_this = self.get_feats_this(idx_this, all_size, vox_ind_gather, is_ind=True)
            
            multi_scale_3d_features_this[key] = spconv.SparseConvTensor(
            features=vox_feats_this,
            indices=vox_ind_this.int(),
            spatial_shape=multi_scale_3d_features[key].spatial_shape,
            batch_size=batch_dict['batch_size']
            )


        batch_dict['multi_scale_3d_features'] = multi_scale_3d_features_this

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

        cluster_ids = batch_dict['cluster_ids'] #original (B=2, 20000)

        # unshuffle point features, points (not needed), batch size bcz no BN in projection layers
        idx_unshuffle = batch_dict.get('idx_unshuffle', None)
        if torch.distributed.is_initialized() and idx_unshuffle is not None:
            batch_dict = self._batch_unshuffle_ddp(batch_dict, idx_unshuffle)

        backbone_3d_bev_feats = batch_dict['spatial_features'] # (6,256,188,188) # unshuffled (B=2, C=128, N num points = 20000)
        backbone_2d_bev_feats = batch_dict['spatial_features_2d'] # (6,512,188,188)
        bev_features = torch.cat([backbone_3d_bev_feats, backbone_2d_bev_feats], dim=1)  # (6,256+512,188,188)
        points = batch_dict['points'] #xyzi
        batch_size = batch_dict["batch_size"] # unshuffled

        batch_seg_feats = []
        for pc_idx in range(batch_size):
            pc = points[pc_idx]
            cluster_labels_this_pc = cluster_ids[pc_idx] # (20000,)
            common_cluster_labels_this_pc = batch_dict['common_cluster_ids'][pc_idx]
            fg_pts_mask = cluster_labels_this_pc > -1
            fg_points = pc[fg_pts_mask][:,0:3].unsqueeze(dim=0) #(1, num fg pts, 3=xyz)

            # Sample fg points
            cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    fg_points[:, :, 0:3].contiguous(), self.NUM_KEYPOINTS
                ).long()# (1, 4096) sample 4096 points from this pc
            

            if fg_points.shape[1] < self.NUM_KEYPOINTS:
                times = int(self.NUM_KEYPOINTS / fg_points.shape[1]) + 1
                non_empty = cur_pt_idxs[0, :fg_points.shape[1]]
                cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

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