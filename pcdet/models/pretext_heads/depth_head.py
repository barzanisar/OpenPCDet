import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DepthHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size):
        super().__init__()
        self.use_mlp = False
        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head
    
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
        idx_unshuffle = batch_dict.get('idx_unshuffle', None)
        if torch.distributed.is_initialized() and idx_unshuffle is not None:
            point_features = self._batch_unshuffle_ddp(point_features, idx_unshuffle)
        
        nump = point_features.shape[-1] # 16384
        
        # get one feature vector of dim 128 for the entire point cloud
        feat = torch.squeeze(F.max_pool1d(point_features, nump)) # (8,128)
        if self.use_mlp:
            feat = self.head(feat)
        batch_dict["pretext_head_feats"] = feat #(B=8, C=128)

        # Convert point features in OpenPCDet format
        point_features = point_features.permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1]) # (16384x2, 128)

        return batch_dict        