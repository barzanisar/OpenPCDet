import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np

from datasets.collators.sparse_collator import list_segments_points, numpy_to_sparse_tensor
from third_party.OpenPCDet.pcdet.models.pretext_heads.mlp import MLP
from third_party.OpenPCDet.pcdet.models.pretext_heads.gather_utils import *


class SegSparseVoxHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size):
        super().__init__()
        self.cluster = True #TODO implement both depth contrast and seg contrast here
        self.use_mlp = False
        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head linear, relu, linear

        self.dropout = ME.MinkowskiDropout(p=0.4)
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, batch_dict, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """ #x is a sparse tensor C:(8pcs x 20k pts, 4=b_id,xyz vox coord) F:(8x20k, 4=xyzi pts)
        # gather from all gpus
        x = batch_dict['sparse_point_feats']
        batch_size = []

        # sparse tensor should be decomposed
        c, f = x.decomposed_coordinates_and_features # c=list of bs=8 [(20k, 3=xyz vox coord), ...,()]
        # f=list of bs=8 [(20k, 4=xyzi pts), ...,()]
        # each pcd has different size, get the biggest size as default
        newx = list(zip(c, f))# list of 8=[(c,f)..., (c,f)]
        for bidx in newx:
            batch_size.append(len(bidx[0]))
        all_size = concat_all_gather(torch.tensor(batch_size).cuda()) # if 2 gpus then list of 16 = [20k, ..., 20k]
        max_size = torch.max(all_size) # max num pts in any pc 20k

        # create a tensor with shape (batch_size, max_size)
        # copy each sparse tensor data to the begining of the biggest sized tensor
        shuffle_c = []
        shuffle_f = []
        for bidx in range(len(newx)):
            shuffle_c.append(torch.ones((max_size, newx[bidx][0].shape[-1])).cuda())
            shuffle_c[bidx][:len(newx[bidx][0]),:] = newx[bidx][0]

            shuffle_f.append(torch.ones((max_size, newx[bidx][1].shape[-1])).cuda())
            shuffle_f[bidx][:len(newx[bidx][1]),:] = newx[bidx][1]

        batch_size_this = len(newx) # 8

        shuffle_c = torch.stack(shuffle_c) # (bs=8, max num pts=20k, 3)
        shuffle_f = torch.stack(shuffle_f) # (8, max num pts=20k, 4)

        # gather all the ddp batches pcds
        c_gather = concat_all_gather(shuffle_c) #(all_bs = 16, max_num_pts=20k, 3)
        f_gather = concat_all_gather(shuffle_f) #(all_bs = 16, max_num_pts=20k, 4)

        batch_size_all = c_gather.shape[0] # 16

        num_gpus = batch_size_all // batch_size_this # 16/8=2

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx] # indices of shuffled pc id [4,6,5,1,.,0,....] -> [[4,6,5,1,.,0,.],[ second 8 indices]] -> pick for this gpu [4,6,5,1,.,0,.,.]

        c_this = []
        f_this = []

        # after unshuffling we get only the actual information of each tensor
        # :actual_size is the information, actual_size:biggest_size are just ones (ignore)
        for idx in range(len(idx_this)):
            c_this.append(c_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())
            f_this.append(f_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())

        # final unshuffled coordinates and features, build back the sparse tensor
        c_this = np.array(c_this) # original pcs are back (8, 20k, 3=xyz vox coord)
        f_this = np.array(f_this) # original pcs are back (8, 20k, 4=xyzi pts)
        x_this = numpy_to_sparse_tensor(c_this, f_this) # sparse tensor in this gpu (C: (8,20k,4=b_id,xyz vox coords), F:(8, 20k, 4=xyzi pts))

        batch_dict['sparse_point_feats'] = x_this
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

        x = batch_dict['sparse_point_feats'] #unshuffled
        cluster_ids = batch_dict['cluster_ids']

        if self.cluster:
            b = list_segments_points(x.C, x.F, np.asarray(cluster_ids)) #[num points in all segments of all 8 pcs x 96]

        # from input points dropout some (increase randomness)
        x = self.dropout(b) #x is a sparse tensor:(C:(numpts in all segs, 4=bidx of seg, xyz vox coord), F:(numpts on all segs,96))[num points belonging to segments x 96]

        # global max pooling over the remaining points
        x = self.glob_pool(x) #[num segments x 96] max pools pts in each seg

        # project the max pooled features
        out = self.head(x.F) #[num segments x 96] -> [num segments x 128]

        batch_dict["pretext_head_feats"] = out

        return batch_dict

 