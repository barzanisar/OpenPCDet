import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from third_party.OpenPCDet.pcdet.models.pretext_heads.mlp import MLP
from third_party.OpenPCDet.pcdet.models.pretext_heads.gather_utils import *


class ProjectionPointHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size):
        super().__init__()
        self.use_mlp = False
        self.cluster = model_cfg.cluster
        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, batch_dict, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus

        points = batch_dict['points'] # unshuffle this (B=2, 20000, 4) needed to create 'point_coords'
        point_features = batch_dict['point_features'] #  unshuffle this (B=2, C=128, N num points = 20000)    
        batch_size_this = batch_dict['batch_size']

        ################################ Unshuffle Points ###########################3
        num_pts_batch = np.unique(points[:,0].cpu().numpy(), return_counts=True)[1]
        all_size = concat_all_gather(torch.tensor(num_pts_batch).cuda())
        max_size = torch.max(all_size) #max num voxels in any pc
        points_gather = gather_feats(batch_indices=points[:,0], 
                                        feats_or_coords=points, 
                                        batch_size_this=batch_size_this, 
                                        num_vox_or_pts_batch=num_pts_batch, 
                                        max_num_vox_or_pts=max_size)

        batch_size_all = points_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx] # indices of shuffled pc id [4,6,5,1,.,0,....] -> [[4,6,5,1,.,0,.],[ second 8 indices]] -> pick for this gpu [4,6,5,1,.,0,.,.]

        batch_dict['batch_size'] = len(idx_this)
        batch_dict['points'] = get_feats_this(idx_this, all_size, points_gather, is_ind=True) # (N1+..Nbs, bxyzi)
   
        ################################ Unshuffle point features ###########################3

        # gather all point features
        all_point_features = concat_all_gather(point_features) #(B=2, C=128, N num points = 20000) -> (8, 128, 20k)
        batch_dict['point_features'] = all_point_features[idx_this] # (2, 128, 20k)
        
        return batch_dict

    
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
        
        point_features = batch_dict['point_features'] # unshuffled (B=2, C=128, N num points = 20000)    
        batch_size = batch_dict["batch_size"] # unshuffled

        if self.cluster:
            #SegContrast
            cluster_ids = batch_dict['cluster_ids'] #original (B=2, 20000)

            batch_seg_feats = []
            for pc_idx in range(batch_size):
                pc_feats = point_features[pc_idx] #(128, 20000)
                cluster_labels_this_pc = cluster_ids[pc_idx] # (20000,)
                common_cluster_labels_this_pc = batch_dict['common_cluster_ids'][pc_idx]
                
                for segment_lbl in common_cluster_labels_this_pc: #np.unique(cluster_labels_this_pc):
                    assert segment_lbl != -1
                    # if segment_lbl == -1:
                    #     continue
                    
                    seg_feats = pc_feats[:,cluster_labels_this_pc == segment_lbl] #(128, npoints in this seg)
                    #seg_feats = self.dout(seg_feats) #zero some values in [128, num points in this cluster]
                    seg_feats = seg_feats.unsqueeze(0) #1,128, npoints in this seg
                    npoints = seg_feats.shape[-1]
                    seg_max_feat = F.max_pool1d(seg_feats, npoints).squeeze(-1) #[1, 128, npoints] -> [1, 128, 1] -> [1, 128]
                    batch_seg_feats.append(seg_max_feat) # (1, 128)
                
            
            feats = torch.vstack(batch_seg_feats) # (num clusters, 128)
        else:
            #DepthContrast
            numpts = point_features.shape[-1] # 16384
            # get one feature vector of dim 128 for the entire point cloud
            feats = torch.squeeze(F.max_pool1d(point_features, numpts)) # (num pc =8,128)            
        
        if self.use_mlp:
            feats = self.head(feats) # (num clusters, 128)
        

        # Convert point features in OpenPCDet format
        batch_dict["pretext_head_feats"] = feats # (num clusters, 128)
        point_features = point_features.permute(0, 2, 1).contiguous()  # (B=2, N=20000, C=128)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1]) # (B x N, 128)
        batch_dict['point_coords'] = batch_dict['points'][:,:4] #(b id, xyz)

        return batch_dict        