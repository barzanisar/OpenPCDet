import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np

from datasets.collators.sparse_collator import list_segments_points, numpy_to_sparse_tensor
from third_party.OpenPCDet.pcdet.models.pretext_heads.mlp import MLP
from third_party.OpenPCDet.pcdet.models.pretext_heads.gather_utils import *
# from third_party.OpenPCDet.pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils

# from torch import einsum
# from einops import repeat

# def exists(val):
#     return val is not None

# def max_value(t):
#     return torch.finfo(t.dtype).max

# class ProposalEncodingLayerV2(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         pos_mlp_hidden_dim = 64,
#         attn_mlp_hidden_mult = 4,
#         downsample = 4,
#     ):
#         super().__init__()

#         self.inter_channels = dim // downsample

#         self.g = nn.Linear(dim, self.inter_channels, bias=False)
#         self.theta = nn.Linear(dim, self.inter_channels, bias=False)
#         self.phi = nn.Linear(dim, self.inter_channels, bias=False)

#         self.pos_mlp = nn.Sequential(
#             nn.Linear(3, pos_mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(pos_mlp_hidden_dim, self.inter_channels)
#         )

#         self.attn_mlp = nn.Sequential(
#             nn.Linear(self.inter_channels, self.inter_channels * attn_mlp_hidden_mult),
#             nn.ReLU(),
#             nn.Linear(self.inter_channels * attn_mlp_hidden_mult, self.inter_channels),
#         )

#         self.conv_out = nn.Linear(self.inter_channels, dim, bias=False)


#     def forward(self, x, pos, mode='cross', mask = None):

#         x, y = x[0], x[1] #(N,1,512) (N, 16, 512)
#         x_pos, y_pos = pos #(N proposals, 1, 3) (N, 16, 3)

#         v = self.g(y) #(N, 16, 512) -> (N, 16, 128)
#         q = self.theta(x) #(N, 1, 512) -> (N, 1, 128)
#         k = self.phi(y) #(N, 16, 512) -> (N, 16, 128)

#         # calculate relative positional embeddings
#         rel_pos = x_pos[:, :, None, :] - y_pos[:, None, :, :] #(N,1,1,3) - (N, 1, 16, 3)
#         rel_pos_emb = self.pos_mlp(rel_pos) #(N, 1, 16, 3) -> (N, 1, 16, 128)

#         # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
#         qk_rel = q[:, :, None, :] - k[:, None, :, :]  #(N, 1, 16, 512)

#         # prepare mask
#         if exists(mask):
#             mask = mask[:, :, None] * mask[:, None, :]

#         # expand values
#         v = repeat(v, 'b j d -> b i j d', i=1) # (N, 1, 16, 128)

#         # add relative positional embeddings to value
#         v = v + rel_pos_emb #  (N, 1, 16, 128)

#         # use attention mlp, making sure to add relative positional embedding first
#         sim = self.attn_mlp(qk_rel + rel_pos_emb)#  (N, 1, 16, 128)

#         # masking
#         if exists(mask):
#             mask_value = -max_value(sim)
#             sim.masked_fill_(~mask[..., None], mask_value)

#         # attention
#         attn = sim.softmax(dim=-2)

#         # aggregate
#         agg = einsum('b i j d, b i j d -> b i d', attn, v)

#         return x + self.conv_out(agg)


class ProjectionSparseVoxHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size):
        super().__init__()
        self.cluster = model_cfg.cluster
        self.use_mlp = False
        if model_cfg.use_mlp:
            self.use_mlp = True
            self.head = MLP(model_cfg.mlp_dim) # projection head linear, relu, linear

        self.dropout = ME.MinkowskiDropout(p=0.4)
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.attn = None
        # if model_cfg.get("proposal_attn_encoder", False):
        #     self.attn = ProposalEncodingLayerV2(
        #     dim=96,
        #     pos_mlp_hidden_dim=64,
        #     attn_mlp_hidden_mult=2,
        #     downsample=2)
        

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, batch_dict, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """ #x is a sparse tensor C:(8pcs x 20k pts, 4=b_id,xyz vox coord) F:(8x20k, 4=xyzi pts)
        # gather from all gpus
        x = batch_dict['sparse_point_feats']
        # point_coords = torch.cat((batch_dict['sparse_points'].C[:,0].unsqueeze(-1), batch_dict['sparse_points'].F[:,:3]), dim=1) #bid, xyz
        num_pts_batch = []

        # sparse tensor should be decomposed
        c, f = x.decomposed_coordinates_and_features # c=list of bs=8 [(20k, 3=xyz vox coord), ...,()]
        # f=list of bs=8 [(20k, 4=xyzi pts), ...,()]
        # each pcd has different size, get the biggest size as default
        newx = list(zip(c, f))# list of 8=[(c,f)..., (c,f)]
        for bidx in newx:
            num_pts_batch.append(len(bidx[0]))
        all_size = concat_all_gather(torch.tensor(num_pts_batch).cuda()) # if 2 gpus then list of 16 = [20k, ..., 20k]
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
        # c_this = np.array(c_this) # original pcs are back (8, 20k, 3=xyz vox coord)
        # f_this = np.array(f_this) # original pcs are back (8, 20k, 4=xyzi pts)
        x_this = numpy_to_sparse_tensor(c_this, f_this) # sparse tensor in this gpu (C: (8,20k,4=b_id,xyz vox coords), F:(8, 20k, 4=xyzi pts))

        batch_dict['sparse_point_feats'] = x_this

        ####################### Unshuffle points #######################
        # points_gather = gather_feats(batch_indices=point_coords[:,0], 
        #                                 feats_or_coords=point_coords, 
        #                                 batch_size_this=batch_size_this, 
        #                                 num_vox_or_pts_batch=num_pts_batch, 
        #                                 max_num_vox_or_pts=max_size)

        batch_dict['batch_size'] = len(idx_this)
        # batch_dict['point_coords'] = get_feats_this(idx_this, all_size, points_gather, is_ind=True) # (N1+..Nbs, bxyzi)
   
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

            # if self.attn is not None:
            #     sampled_batch_seg_pts = []
            #     sampled_batch_seg_pt_feats = []
            #     batch_seg_centers = []
            #     batch_seg_center_feats =[]
            #     for batch_num in range(len(cluster_ids)):
            #         num_gt_boxes = len(batch_dict['gt_boxes_cluster_ids'][batch_num])
            #         gt_box_centers = batch_dict['gt_boxes'][batch_num, :num_gt_boxes, :3]
            #         gt_box_cluster_ids = batch_dict['gt_boxes_cluster_ids'][batch_num]
            #         unique_cluster_ids = np.unique(cluster_ids[batch_num])
            #         assert gt_box_centers.shape[0] == gt_box_cluster_ids.shape[0]

            #         gt_box_idx_of_unique_clusters=np.where(np.isin(gt_box_cluster_ids, unique_cluster_ids))[0]
                    
            #         #sanity check that unique cluster lbls match with the gt box idx found
            #         # gt_box_idx_of_unique_clusters_list1 = []
            #         # for lbl in np.unique(cluster_ids[batch_num]):
            #         #     if lbl == -1:
            #         #         continue
            #         #     gt_box_idx_for_this_lbl = np.where(gt_box_cluster_ids==lbl)[0][0]
            #         #     gt_box_idx_of_unique_clusters_list1.append(gt_box_idx_for_this_lbl)

            #         # assert (gt_box_idx_of_unique_clusters_list1 == gt_box_idx_of_unique_clusters).all()
            #         # i = 0
            #         # for lbl in unique_cluster_ids:
            #         #     if lbl == -1:
            #         #         continue
            #         #     assert gt_box_cluster_ids[gt_box_idx_of_unique_clusters[i]] == lbl   
            #         #     i +=1
                    
            #         gt_box_centers_of_unique_clusters = gt_box_centers[gt_box_idx_of_unique_clusters]

            #         b_mask = x.C[:,0] == batch_num
            #         pts_xyz = batch_dict['point_coords'][b_mask][:,1:]
            #         pts_feats = x.F[b_mask, :]
                    
            #         dist, idx = pointnet2_utils.three_nn(gt_box_centers_of_unique_clusters.unsqueeze(0).contiguous(), pts_xyz.unsqueeze(0).contiguous())
            #         dist_recip = 1.0 / (dist + 1e-8)
            #         norm = torch.sum(dist_recip, dim=2, keepdim=True)
            #         weight = dist_recip / norm
            #         interpolated_feats_at_centers = pointnet2_utils.three_interpolate(pts_feats.transpose(1,0).unsqueeze(0).contiguous(), idx, weight) #(num gt centers, C)

            #         batch_seg_centers.append(gt_box_centers_of_unique_clusters)
            #         batch_seg_center_feats.append(interpolated_feats_at_centers.squeeze(0).transpose(1,0))

            #         for segment_lbl in unique_cluster_ids:
            #             if segment_lbl == -1:
            #                 continue
                        
            #             seg_mask = cluster_ids[batch_num] == segment_lbl
            #             fps_choice = pointnet2_utils.furthest_point_sample(pts_xyz[seg_mask].unsqueeze(0).contiguous(), 16).long().squeeze()
            #             sampled_batch_seg_pts.append(pts_xyz[seg_mask][fps_choice])
            #             sampled_batch_seg_pt_feats.append(pts_feats[seg_mask][fps_choice])
                    
            #     sampled_batch_seg_pts = torch.stack(sampled_batch_seg_pts) # (N segs, 16 pts each seg, 3)
            #     sampled_batch_seg_pt_feats = torch.stack(sampled_batch_seg_pt_feats) # (N segs, 16 pts each seg, 96)
            #     batch_seg_centers = torch.cat(batch_seg_centers, dim=0).unsqueeze(1) #(N segs, 1, 3)
            #     batch_seg_center_feats = torch.cat(batch_seg_center_feats, dim=0).unsqueeze(1) #(N segs, 1, 96)
            #     input_xyz = (batch_seg_centers, sampled_batch_seg_pts)
            #     input_features = (batch_seg_center_feats, sampled_batch_seg_pt_feats)
            #     x = self.attn(input_features, input_xyz).squeeze() #[num segments x 96]
            #     out = self.head(x)
            # else:
            x = list_segments_points(x.C, x.F, cluster_ids) #[num points in all segments of all 8 pcs x 96]

        x = self.dropout(x) #x is a sparse tensor:(C:(numpts in all segs, 4=bidx of seg, xyz vox coord), F:(numpts on all segs,96))[num points belonging to segments x 96]

        # global max pooling over the remaining points
        x = self.glob_pool(x) #[num segments x 96] max pools pts in each seg

        # project the max pooled features
        out = self.head(x.F) #[num segments x 96] -> [num segments x 128]

        batch_dict["pretext_head_feats"] = out
        batch_dict['point_features'] = batch_dict['sparse_point_feats'].F
        # if 'point_coords' not in batch_dict:
        #     batch_dict['point_coords'] =  torch.cat((batch_dict['sparse_points'].C[:,0].unsqueeze(-1), batch_dict['sparse_points'].F[:,:3]), dim=1) #bid, xyz

        return batch_dict

 