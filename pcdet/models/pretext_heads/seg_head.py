import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from third_party.OpenPCDet.pcdet.models.pretext_heads.mlp import MLP


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


class SegHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size):
        super().__init__()
        self.use_mlp = False
        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head

        #self.dout=nn.Dropout(p=0.3)
    
    @torch.no_grad()
    def gather_feats(self, batch_indices, feats_or_coords, batch_size_this, num_vox_or_pts_batch, max_num_vox_or_pts):
        shuffle_feats=[]
        shuffle_feats_shape = list(feats_or_coords.size())
        shuffle_feats_shape[0] = max_num_vox_or_pts.item()

        for bidx in range(batch_size_this):
            num_vox_this_pc = num_vox_or_pts_batch[bidx]
            b_mask = batch_indices == bidx

            shuffle_feats.append(torch.ones(shuffle_feats_shape).cuda())
            shuffle_feats[bidx][:num_vox_this_pc] =  feats_or_coords[b_mask]


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

        points = batch_dict['points'] # unshuffle this (B=2, 20000, 4) needed to create 'point_coords'
        point_features = batch_dict['point_features'] #  unshuffle this (B=2, C=128, N num points = 20000)    
        batch_size_this = batch_dict['batch_size']

        ################################ Unshuffle Points ###########################3
        num_pts_batch = np.unique(points[:,0].cpu().numpy(), return_counts=True)[1]
        all_size = concat_all_gather(torch.tensor(num_pts_batch).cuda())
        max_size = torch.max(all_size) #max num voxels in any pc
        points_gather = self.gather_feats(batch_indices=points[:,0], 
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
        batch_dict['points'] = self.get_feats_this(idx_this, all_size, points_gather, is_ind=True) # (N1+..Nbs, bxyzi)
   
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

        cluster_ids = batch_dict['cluster_ids'] #original (B=2, 20000)

        # unshuffle point features, points (not needed), batch size bcz no BN in projection layers
        idx_unshuffle = batch_dict.get('idx_unshuffle', None)
        if torch.distributed.is_initialized() and idx_unshuffle is not None:
            batch_dict = self._batch_unshuffle_ddp(batch_dict, idx_unshuffle)

        point_features = batch_dict['point_features'] # unshuffled (B=2, C=128, N num points = 20000)    
        batch_size = batch_dict["batch_size"] # unshuffled

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
            
        
        all_seg_feats = torch.vstack(batch_seg_feats) # (num clusters, 128)
        if self.use_mlp:
            all_seg_feats = self.head(all_seg_feats) # (num clusters, 128)
        

        # Convert point features in OpenPCDet format
        batch_dict["pretext_head_feats"] = all_seg_feats # (num clusters, 128)
        point_features = point_features.permute(0, 2, 1).contiguous()  # (B=2, N=20000, C=128)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1]) # (B x N, 128)
        batch_dict['point_coords'] = batch_dict['points'][:,:4] 

        return batch_dict        