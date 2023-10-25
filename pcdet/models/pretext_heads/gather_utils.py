import torch

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
    

@torch.no_grad()
def gather_feats(batch_indices, feats_or_coords, batch_size_this, num_vox_or_pts_batch, max_num_vox_or_pts):
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
def get_feats_this(idx_this, all_size, gather_feats, is_ind=False):
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