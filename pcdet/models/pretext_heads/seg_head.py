import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pcdet.models.pretext_heads.mlp import MLP


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
        cluster_ids = batch_dict['cluster_ids']
        batch_size = batch_dict["batch_size"]

        idx_unshuffle = batch_dict.get('idx_unshuffle', None)
        if torch.distributed.is_initialized() and idx_unshuffle is not None:
            point_features = self._batch_unshuffle_ddp(point_features, idx_unshuffle)


        batch_seg_feats = []
        for pc_idx in range(batch_size):
            pc_feats = point_features[pc_idx] #(128 x 16384)
            cluster_labels_this_pc = cluster_ids[pc_idx] # (16384,)
            for segment_lbl in np.unique(cluster_labels_this_pc):
                if segment_lbl == -1:
                    continue

                seg_feats = pc_feats[:,cluster_labels_this_pc == segment_lbl] #(128, npoints in this seg)
                seg_feats = self.dout(seg_feats) #zero some values in [128, num points in this cluster]
                seg_feats = seg_feats.unsqueeze(0) #1,128, npoints in this seg
                npoints = seg_feats.shape[-1]
                seg_max_feat = F.max_pool1d(seg_feats, npoints).squeeze(-1) #[1, 128, npoints] -> [1, 128, 1] -> [1, 128]
                batch_seg_feats.append(seg_max_feat)
        
        all_seg_feats = torch.vstack(batch_seg_feats) # num clusters x 128
        if self.use_mlp:
            all_seg_feats = self.head(all_seg_feats) # num clusters x 128
        

        # Convert point features in OpenPCDet format
        batch_dict["pretext_head_feats"] = all_seg_feats #(B=8, C=128)
        point_features = point_features.permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1]) # (16384x2, 128)

        return batch_dict        