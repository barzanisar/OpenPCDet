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
    def __init__(self, model_cfg):
        super().__init__()
        self.use_mlp = False
        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head

        self.dout=nn.Dropout(p=0.3)
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, batch_dict, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus

        points = batch_dict['points'] # unshuffle this (Bx20000, 4) needed to create 'point_coords'
        point_features = batch_dict['point_features'] #  unshuffle this (B=2, C=128, N num points = 20000)    
        batch_size_this = batch_dict['batch_size'] #  unshuffle this

        # # gather all pcs
        all_pcs = concat_all_gather(points) # (2, 20000, 4) -> (8, 20000, 4)

        # gather all point features
        all_point_features = concat_all_gather(point_features) #(B=2, C=128, N num points = 20000) -> (8, 128, 20k)
        
        batch_size_all = all_point_features.shape[0]
        num_gpus = batch_size_all // batch_size_this
        
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        
        # No need to unshuffle points because they are not needed.
        batch_dict['point_features'] = all_point_features[idx_this] # (2, 128, 20k)
        batch_dict['batch_size'] = batch_dict['point_features'].shape[0]
        batch_dict['points'] = all_pcs[idx_this] #(B, 20k, 4) 
        
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

        cluster_ids = batch_dict['box_ids_of_pts'] #original (B=2, 20000)

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
            
            for segment_lbl in np.unique(cluster_labels_this_pc):
                if segment_lbl == -1:
                    continue
                
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
        num_points_per_pc = batch_dict['points'].shape[1]
        
        batch_ids = torch.stack([batch_idx*torch.ones(num_points_per_pc) for batch_idx in range(batch_dict['batch_size'])])
        batch_ids = batch_ids.to(batch_dict['points'], non_blocking=True).unsqueeze(-1)
        point_coords = torch.cat([batch_ids, batch_dict['points'][:,:,:3]], dim=2) #batch_idx, xyz
        batch_dict['point_coords'] = point_coords.view(-1, 4) # (B, 20K, 4) -> #(BxN, 4) batch_idx, xyz
        points_feature_dim = batch_dict['points'].shape[-1]
        batch_dict['points'] = batch_dict['points'].view(-1, points_feature_dim) # (BxN, 4) xyzi

        return batch_dict        